"""Phone -> laptop camera/mic link, isolated, with measurements: is the link good enough (goal: <= ~50 ms, >= 25 fps,
steady)? The phone page encodes its camera with the hardware H.264 encoder (WebCodecs, ~3 Mbit/s instead of ~30 for
JPEGs) and stamps every frame with its capture time; the laptop syncs clocks with the phone (NTP-style pings over the
same socket, min-RTT sample) and measures per frame: phone capture -> laptop decoded (the link latency, excluding
the camera sensor itself), frame rate, arrival gaps, dropped frames, bitrate; and the mic's latency the same way.

  cloud/facefix/.venv/bin/python tools/phone_link/phone_link.py [--iface wlx3460f993bc3d] [--port 8446]
  -> prints the phone URL (HTTPS, self-signed: accept once); laptop view + live numbers: http://127.0.0.1:8782
  per-frame log: .tmp/phone_link/frames.csv

Nothing here talks to the internet (the page loads no external resources, no WebRTC/STUN).
"""
import argparse
import asyncio
import collections
import http
import json
import secrets
import struct
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import av
import cv2
import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

REPO = Path(__file__).resolve().parents[2]
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(REPO / "tools"))
from phone_cam_js import CAM_CSS, CAM_JS  # noqa: E402
OUT = REPO / ".tmp" / "phone_link"
HDR = struct.Struct("<dIB")      # capture time (epoch ms, phone clock), seq, keyframe flag


def iface_ips():
    rows = [ln.split() for ln in subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True,
                                                 text=True).stdout.splitlines()]
    return {r[1]: r[3].split("/")[0] for r in rows if len(r) > 3 and not r[3].startswith("127.")
            and not r[1].startswith(("docker", "br-", "veth"))}


def cert(ips):
    """Signed by the local CA (tools/local_ca.py): once the phone trusts the CA, pages AND WebSockets just work."""
    import ssl
    import sys as _s
    _s.path.insert(0, str(REPO / "tools"))
    from local_ca import server_cert
    crt, key = server_cert(ips, OUT)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    return ctx


def cert_selfsigned(ips):
    import ssl
    OUT.mkdir(parents=True, exist_ok=True)
    crt, key, tag = OUT / "cert.pem", OUT / "key.pem", OUT / "ips.txt"
    want = ",".join(sorted(ips))
    if not crt.exists() or not tag.exists() or tag.read_text() != want:
        san = ",".join([f"IP:{i}" for i in ips] + ["DNS:localhost"])
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825", "-keyout", str(key),
                        "-out", str(crt), "-subj", "/CN=phone-link", "-addext", f"subjectAltName={san}"],
                       check=True, capture_output=True)
        tag.write_text(want)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    return ctx


class Clock:
    """Phone clock -> laptop clock: offset from the ping with the smallest round trip in the last ~30 s."""

    def __init__(self):
        self.samples = collections.deque(maxlen=60)     # (rtt, offset, t)

    def add(self, t_send, t_phone, t_recv):
        rtt = t_recv - t_send
        self.samples.append((rtt, t_phone - (t_send + rtt / 2), t_recv))

    def offset(self):
        if not self.samples:
            return None
        return min(self.samples)[1]

    def rtt(self):
        return min(self.samples)[0] if self.samples else None


class Link:
    def __init__(self, a):
        self.a, self.clock, self.ws = a, Clock(), None
        self.dec = av.CodecContext.create("h264", "r")
        self.latest_jpg = None
        self.lat = collections.deque(maxlen=250)        # video: capture -> decoded on the laptop (ms)
        self.alat = collections.deque(maxlen=250)       # mic: capture -> arrived (ms)
        self.arr = collections.deque(maxlen=250)        # arrival times (s)
        self.dec_ms = collections.deque(maxlen=250)
        self.bytes = collections.deque(maxlen=250)      # (t, n)
        self.last_seq, self.dropped, self.frames, self.kind = None, 0, 0, None
        self.phone = {}
        self.http_last = self.http_ping = 0.0
        self.csv = open(OUT / "frames.csv", "w")
        self.csv.write("t_arrival_ms,seq,key,bytes,latency_ms,decode_ms,rtt_ms\n")

    async def pinger(self, ws):
        while True:
            await ws.send(b"P" + struct.pack("<d", time.time() * 1000))
            await asyncio.sleep(0.5)

    def video(self, m, jpeg=False):
        t_arr = time.time() * 1000
        cap, seq, flags = HDR.unpack_from(m, 1)
        key, rot = flags & 1, (flags >> 1) & 3
        data = m[1 + HDR.size:]
        self.kind = "jpeg" if jpeg else "h264"
        if self.last_seq is not None and seq > self.last_seq + 1:
            self.dropped += seq - self.last_seq - 1
        self.last_seq = seq
        t0 = time.perf_counter()
        img = None
        try:
            if jpeg:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            else:
                for fr in self.dec.decode(av.Packet(data)):
                    img = fr.to_ndarray(format="bgr24")
                if img is not None and rot:   # the phone sends sensor-oriented frames: upright here (~0.5 ms)
                    img = cv2.rotate(img, (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE)[rot - 1])
        except Exception:  # noqa: BLE001  (a missing keyframe after a reconnect: wait for the next one)
            img = None
        dms = (time.perf_counter() - t0) * 1000
        t_done = time.time() * 1000
        off = self.clock.offset()
        lat = None if off is None else t_done - (cap - off)
        self.frames += 1
        self.arr.append(t_arr / 1000)
        self.bytes.append((t_arr / 1000, len(data)))
        self.dec_ms.append(dms)
        if lat is not None:
            self.lat.append(lat)
        self.csv.write(f"{t_arr:.1f},{seq},{key},{len(data)},{'' if lat is None else f'{lat:.1f}'},{dms:.2f},"
                       f"{'' if self.clock.rtt() is None else f'{self.clock.rtt():.1f}'}\n")
        if img is not None and self.frames % 2 == 0:      # laptop preview at ~15 fps
            ok, j = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                self.latest_jpg = j.tobytes()

    def audio(self, m):
        cap = struct.unpack_from("<d", m, 1)[0]
        off = self.clock.offset()
        if off is not None:
            self.alat.append(time.time() * 1000 - (cap - off))

    async def handler(self, ws):
        if not ws.request.path.startswith(f"/ws?t={self.a.token}"):
            await ws.close(1008, "bad token")
            return
        if self.ws is not None:
            await self.ws.close(1000, "replaced by a newer connection")
        self.ws, self.dec = ws, av.CodecContext.create("h264", "r")
        self.last_seq = None
        print("[link] phone connected", flush=True)
        pt = asyncio.create_task(self.pinger(ws))
        try:
            async for m in ws:
                if isinstance(m, bytes) and m:
                    self.on_msg(m)
        except Exception:  # noqa: BLE001
            import traceback
            print("[link] connection error:\n" + traceback.format_exc(), flush=True)
        finally:
            pt.cancel()
            if self.ws is ws:
                self.ws = None
            print("[link] phone disconnected", flush=True)

    def on_msg(self, m):
        k = m[:1]
        if k == b"H":
            self.video(m)
        elif k == b"J":
            self.video(m, jpeg=True)
        elif k == b"A":
            self.audio(m)
        elif k == b"Q":
            t_send, t_phone = struct.unpack_from("<dd", m, 1)
            self.clock.add(t_send, t_phone, time.time() * 1000)
        elif k == b"S":
            try:
                self.phone = {**json.loads(m[1:].decode()), "t": time.time()}
            except ValueError:
                pass

    async def http_uplink(self, r, w):
        """Fallback transport when Safari's WebSocket won't open (it hangs for minutes at times while plain HTTPS
        works): POST /up?t=TOKEN, body = [u32 len][message]..., response = the same framing with what the laptop
        has for the phone (clock pings). Keep-alive; CORS-simple (text/plain) so no preflight."""
        try:
            while True:
                head = await r.readuntil(b"\r\n\r\n")
                lines = head.decode(errors="replace").split("\r\n")
                method, path = (lines[0].split(" ") + ["", ""])[:2]
                hdrs = {ln.split(":", 1)[0].strip().lower(): ln.split(":", 1)[1].strip() for ln in lines[1:] if ":" in ln}
                body = await r.readexactly(int(hdrs.get("content-length", 0) or 0))
                out = b""
                if method == "POST" and path.startswith(f"/up?t={self.a.token}"):
                    if time.time() - self.http_last > 2:
                        print("[link] phone on the HTTP fallback (no WebSocket)", flush=True)
                        self.dec = av.CodecContext.create("h264", "r")
                    self.http_last = time.time()
                    i = 0
                    while i + 4 <= len(body):
                        n = struct.unpack_from("<I", body, i)[0]
                        self.on_msg(body[i + 4:i + 4 + n])
                        i += 4 + n
                    if time.time() - self.http_ping > 0.5:      # clock ping (the phone answers at once)
                        self.http_ping = time.time()
                        p = b"P" + struct.pack("<d", time.time() * 1000)
                        out = struct.pack("<I", len(p)) + p
                    code = "200 OK"
                elif method == "OPTIONS":
                    code = "204 No Content"
                else:
                    code = "404 Not Found"
                w.write((f"HTTP/1.1 {code}\r\nContent-Type: application/octet-stream\r\n"
                         f"Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: *\r\n"
                         f"Access-Control-Allow-Methods: POST\r\nContent-Length: {len(out)}\r\n\r\n").encode() + out)
                await w.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError, asyncio.LimitOverrunError):
            pass
        finally:
            w.close()

    def stats(self):
        now = time.time()
        arr = [t for t in self.arr if now - t < 3]
        gaps = np.diff(arr) * 1000 if len(arr) > 2 else np.array([])
        lat = np.array(self.lat)
        alat = np.array(self.alat)
        br = sum(n for t, n in self.bytes if now - t < 3) * 8 / 3 / 1e6
        pct = lambda x, p: None if not len(x) else round(float(np.percentile(x, p)), 1)  # noqa: E731
        via = "websocket" if self.ws is not None else ("http fallback" if now - self.http_last < 2 else None)
        return {"connected": via is not None, "transport": via, "kind": self.kind,
                "fps": round(len(arr) / 3, 1) if arr else 0.0,
                "latency_ms": {"p50": pct(lat, 50), "p90": pct(lat, 90), "p99": pct(lat, 99), "max": pct(lat, 100)},
                "gap_ms": {"p50": pct(gaps, 50), "p90": pct(gaps, 90), "max": pct(gaps, 100)},
                "audio_latency_ms": {"p50": pct(alat, 50), "p90": pct(alat, 90)},
                "decode_ms": pct(np.array(self.dec_ms), 50), "mbit_s": round(br, 2),
                "dropped": self.dropped, "frames": self.frames,
                "clock_rtt_ms": None if self.clock.rtt() is None else round(self.clock.rtt(), 1),
                "phone": self.phone if self.phone and now - self.phone.get("t", 0) < 5 else {}}

    def http(self, conn, req):
        try:
            peer = conn.remote_address[0]
        except Exception:  # noqa: BLE001
            peer = "?"
        print(f"[link] request {req.path.split('?')[0][:12]}… from {peer}", flush=True)
        if req.path.startswith("/ws"):
            return None
        if req.path.split("?")[0] == "/ca.crt":   # the local CA, for the phone to install + trust once
            from local_ca import CA_DER, ensure_ca
            ensure_ca()
            body = CA_DER.read_bytes()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "application/x-x509-ca-cert"),
                                                               ("Content-Disposition", 'attachment; filename="ai-livestream-local-ca.crt"'),
                                                               ("Content-Length", str(len(body))), ("Connection", "close")]), body)
        p = req.path.split("?")[0]
        if (p == "/" and f"t={self.a.token}" in req.path) or p.rstrip("/") == f"/{self.a.token}":
            body = (PHONE_PAGE.replace("__CAM_JS__", CAM_JS).replace("__CAM_CSS__", CAM_CSS)
                    .replace("__HTTP_PORT__", str(self.a.port + 3))
                    .replace("__TOKEN__", self.a.token).replace("__BITRATE__", str(self.a.bitrate))).encode()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "text/html; charset=utf-8"),
                                                               ("Cache-Control", "no-store"),
                                                               ("Content-Length", str(len(body))), ("Connection", "close")]), body)
        body = b"<h2 style='font-family:system-ui'>Wrong link: open the full link the laptop shows (it ends with a code).</h2>"
        return Response(http.HTTPStatus.NOT_FOUND, "Not Found", Headers([("Content-Type", "text/html"),
                                                                         ("Content-Length", str(len(body))), ("Connection", "close")]), body)


def laptop_server(link, port, urls):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/":
                b = LAPTOP_PAGE.replace("__URLS__", json.dumps(urls)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path == "/stats":
                b = json.dumps(link.stats()).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif self.path.startswith("/video"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
                self.end_headers()
                last = None
                try:
                    while True:
                        j = link.latest_jpg
                        if j is not None and j is not last:
                            last = j
                            self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(j)).encode()
                                             + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    return
            else:
                self.send_response(404)
                self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8446)
    ap.add_argument("--view-port", type=int, default=8782)
    ap.add_argument("--iface", default=None, help="show the URL for this interface first (e.g. the Beryl dongle)")
    ap.add_argument("--bitrate", type=int, default=3_000_000)
    ap.add_argument("--token", default=None)
    a = ap.parse_args()
    a.token = a.token or secrets.token_urlsafe(9)
    ips = iface_ips()
    order = sorted(ips.items(), key=lambda kv: (kv[0] != a.iface, not kv[0].startswith("wl"), kv[0]))
    urls = [(name, f"https://{ip}:{a.port}/{a.token}") for name, ip in order]
    OUT.mkdir(parents=True, exist_ok=True)
    link = Link(a)
    laptop_server(link, a.view_port, urls)
    ctx = cert([ip for _, ip in order])
    await asyncio.start_server(link.http_uplink, "0.0.0.0", a.port + 3, ssl=ctx, limit=16 * 2**20)
    async with serve(link.handler, "0.0.0.0", a.port, ssl=ctx, process_request=link.http,
                     max_size=8 * 2**20, ping_interval=20):
        for name, u in urls:
            print(f"[link] phone URL via {name}: {u}", flush=True)
        print(f"[link] laptop view + numbers: http://127.0.0.1:{a.view_port}", flush=True)
        await asyncio.Future()


PHONE_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Phone link test</title>
<style>[hidden]{display:none!important} body{margin:0;background:#000;color:#ddd;font:15px -apple-system,system-ui}
#me{width:100%;max-height:55vh;object-fit:contain;background:#111}__CAM_CSS__
#p{padding:12px} button{font:600 17px -apple-system;border:0;border-radius:22px;padding:12px 22px;background:#2e7d32;color:#fff}
pre{font:13px ui-monospace,monospace;white-space:pre-wrap}</style></head><body>
<video id="me" playsinline muted autoplay></video>
<div id="p"><button id="go">Start link test</button><div id="cams"></div>
<div id="camrow"><input id="zoom" type="range" hidden><span id="zoomv"></span><button id="rot">↻ rotate</button></div><div id="sock" style="font:13px ui-monospace,monospace;margin:6px 0;color:#e0a03a">socket: –</div><pre id="st">Camera + mic go to the laptop only (local network).</pre></div>
<script>
__CAM_JS__
const T="__TOKEN__", BR=__BITRATE__, $=id=>document.getElementById(id); var cam=null, wsT0=Date.now(), started=false;
const now=()=>performance.timeOrigin+performance.now();
let ws, enc=null, seq=0, nEnc=0, nSent=0, encMs=[], kind="", fpsT=now();
function hdr(tag,cap,sq,key){const b=new ArrayBuffer(14); const v=new DataView(b); v.setUint8(0,tag); v.setFloat64(1,cap,true); v.setUint32(9,sq,true); v.setUint8(13,key===true?1:(key|0)); return new Uint8Array(b);}
// transport: the WebSocket when it's open, else (Safari's WebSocket can hang for minutes) HTTPS POSTs to the laptop
let httpMode=false, outq=[], outBytes=0, inflight=false;
function out(msg){ if(ws&&ws.readyState===1){ if(ws.bufferedAmount>2e6) return false; ws.send(msg); return true; }
  if(!httpMode) return false;
  if(outBytes>2e6) return false;                       // link can't keep up: drop
  outq.push(msg); outBytes+=msg.byteLength; post(); return true; }
async function post(){ if(inflight||!outq.length) return; inflight=true;
  const n=outq.reduce((a,m)=>a+4+m.byteLength,0), b=new Uint8Array(n), dv=new DataView(b.buffer); let i=0;
  for(const m of outq){ dv.setUint32(i,m.byteLength,true); b.set(new Uint8Array(m.buffer||m,m.byteOffset||0,m.byteLength),i+4); i+=4+m.byteLength; }
  outq=[]; outBytes=0;
  try{ const r=await fetch(`https://${location.hostname}:__HTTP_PORT__/up?t=${T}`,{method:"POST",body:b,headers:{"Content-Type":"text/plain"}});
    const a=new Uint8Array(await r.arrayBuffer()), v=new DataView(a.buffer); let j=0;
    while(j+4<=a.length){ const len=v.getUint32(j,true); onDown(new DataView(a.buffer,j+4,len)); j+=4+len; }
    if(httpMode) sock("HTTP fallback — sending to the laptop (WebSocket unavailable)",true);
  }catch(e){ sock("HTTP fallback error: "+e.message); await new Promise(r=>setTimeout(r,500)); }
  inflight=false; if(outq.length) post(); }
function onDown(d){ if(d.getUint8(0)===80){ // 'P' clock ping: echo with our clock, at once
    const o=new Uint8Array(17), v=new DataView(o.buffer); v.setUint8(0,81); v.setFloat64(1,d.getFloat64(1,true),true); v.setFloat64(9,now(),true); out(o); } }
function send(tag,cap,sq,key,data){
  const h=hdr(tag,cap,sq,key), o=new Uint8Array(h.length+data.byteLength); o.set(h,0); o.set(new Uint8Array(data),h.length); if(out(o)) nSent++; }
let nTry=0;
function sock(m,ok){ $("sock").textContent="socket: "+m; $("sock").style.color=ok?"#3fb27f":"#e0a03a"; }
function connect(){ nTry++; wsT0=Date.now(); sock(`connecting (try ${nTry})…`); try{ ws=new WebSocket(`wss://${location.host}/ws?t=${T}`); }catch(e){ sock("can't open: "+e.message); setTimeout(connect,1500); return; }
  ws.binaryType="arraybuffer"; ws.onerror=()=>sock(`error (try ${nTry})`);
  ws.onmessage=e=>onDown(new DataView(e.data));
  ws.onopen=()=>{ httpMode=false; sock("open — sending to the laptop (WebSocket)",true); if(cam) cam.key(); report({hello:navigator.userAgent, webcodecs:"VideoEncoder" in window, rvfc:"requestVideoFrameCallback" in HTMLVideoElement.prototype}); };
  ws.onclose=e=>{ sock(`closed (code ${e.code}${e.reason?": "+e.reason:""}), retrying…`); setTimeout(connect,1000); }; }
function report(o){ out(new TextEncoder().encode("S"+JSON.stringify(o))); }
function fail(where,e){ const m=`${where}: ${e&&e.name?e.name+": ":""}${e&&e.message||e}`; $("st").textContent="ERROR "+m; $("go").hidden=false; report({error:m}); }
window.addEventListener("error",e=>fail("page",e.error||e.message)); window.addEventListener("unhandledrejection",e=>fail("page",e.reason));
connect();   // the socket first: the laptop sees the phone (and any error) even before Start
$("go").onclick=()=>start().then(()=>{try{localStorage.setItem("autostart","1")}catch(_){}}).catch(e=>fail("start",e));
// resilience: a stuck socket is retried, and coming back to the tab (or reopening the page) restores everything
setInterval(()=>{ if(ws&&ws.readyState===0&&Date.now()-wsT0>4000){ try{ws.close()}catch(_){} }
  // no open WebSocket for 3 s: carry on over HTTPS (a keyframe first: the laptop's decoder starts fresh), keep retrying
  if(!(ws&&ws.readyState===1)&&!httpMode&&Date.now()-wsT0>3000){ httpMode=true; if(cam) cam.key(); sock("switching to the HTTP fallback…"); post(); } },1000);
function revive(){ if(document.visibilityState!=="visible") return;
  if(!ws||ws.readyState>1) connect();
  if(cam) cam.ensure(); wake(); }
document.addEventListener("visibilitychange",revive); window.addEventListener("pageshow",revive); window.addEventListener("focus",revive);
let wl=null; async function wake(){ try{ if(!wl||wl.released) wl=await navigator.wakeLock.request("screen"); }catch(_){} }
try{ if(localStorage.getItem("autostart")) setTimeout(()=>{ if(!started) start().catch(()=>{ $("go").hidden=false; sock("tap Start (iOS wants a tap)"); }); },300); }catch(_){}
async function start(){ if(started) return; started=true; $("go").hidden=true; $("st").textContent="starting camera…";
  if(!("VideoEncoder" in window)) throw new Error("this browser has no WebCodecs video encoder");
  kind="h264 (WebCodecs, hardware)";
  cam=CamEngine({video:$("me"), bitrate:BR, onerror:m=>fail("camera",m),
    send:(cap,sq,flags,b)=>send(72,cap,sq,flags,b)});
  try{ await cam.start(); }catch(e){ started=false; $("go").hidden=false; throw e; }
  wake();
  // mic separately (a camera switch never touches it): timestamped PCM (48 kHz s16)
  const s=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1}});
  const ctx=new AudioContext(); await ctx.resume(); const src=ctx.createMediaStreamSource(s), pr=ctx.createScriptProcessor(1024,1,1), r=48000/ctx.sampleRate;
  pr.onaudioprocess=e=>{ if(!ws||ws.readyState!==1)return; const x=e.inputBuffer.getChannelData(0), n=Math.floor(x.length*r), o=new Int16Array(n);
    for(let i=0;i<n;i++)o[i]=Math.max(-1,Math.min(1,x[Math.min(x.length-1,Math.floor(i/r))]))*32767;
    const cap=now()-1000*x.length/ctx.sampleRate, b=new Uint8Array(9+o.byteLength), dv=new DataView(b.buffer); dv.setUint8(0,65); dv.setFloat64(1,cap,true); b.set(new Uint8Array(o.buffer),9); ws.send(b);};
  const z=ctx.createGain(); z.gain.value=0; src.connect(pr); pr.connect(z); z.connect(ctx.destination);
  setInterval(()=>{ const dt=(now()-fpsT)/1000; fpsT=now(); const c=cam.stats();
    const rep={kind, camera:`${c.label} ${c.camera} (rot ${c.rot}°)`, enc_fps:+(c.enc_fps/dt).toFixed(1), sent_fps:+(nSent/dt).toFixed(1), encode_ms:c.encode_ms, ws_buffer:ws?ws.bufferedAmount:0};
    nSent=0; $("st").textContent=Object.entries(rep).map(([k,v])=>`${k}: ${v}`).join("\n");
    report(rep); },1000);
}
</script></body></html>"""

LAPTOP_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Phone Link</title>
<style>body{margin:0;background:#0e1013;color:#e6e8eb;font:14px system-ui;padding:16px} h1{font-size:16px;margin:0 0 10px}
.wrap{display:flex;gap:16px;flex-wrap:wrap} img{height:70vh;max-width:100%;object-fit:contain;background:#000;border-radius:6px} table{border-collapse:collapse}
td{padding:3px 10px;border-bottom:1px solid #2a2f37;font-variant-numeric:tabular-nums} td:first-child{color:#8b939e}
.ok{color:#3fb27f}.warn{color:#e0a03a}.bad{color:#e05a4f} code{color:#5a9cf0;word-break:break-all}</style></head><body>
<h1>Phone link test</h1>
<div style="display:flex;gap:16px;align-items:center;margin-bottom:12px"><div id="qr" style="background:#fff;padding:8px;border-radius:6px;line-height:0"></div>
<div><div style="color:#8b939e">Scan with the phone camera (the phone on the Beryl uses the first link):</div><div id="urls"></div></div></div>
<details style="margin:0 0 12px"><summary style="cursor:pointer;color:#e0a03a">First time on a phone: install the local certificate (once)</summary>
<div style="display:flex;gap:16px;align-items:center;margin-top:8px"><div id="qrca" style="background:#fff;padding:8px;border-radius:6px;line-height:0"></div>
<ol style="margin:0;color:#c8ccd2;line-height:1.6"><li>Scan, open, accept the warning one last time → <b>Allow</b> the download</li>
<li>Settings → <b>Profile Downloaded</b> → Install</li><li>Settings → General → About → <b>Certificate Trust Settings</b> → turn on "ai-livestream local CA (LAN only)"</li>
<li>Valid only for local addresses (192.168.x.x): it can't vouch for any website. Remove any time: Settings → General → VPN &amp; Device Management.</li></ol></div></details>
<div class="wrap"><img id="v" src="/video"><table id="t"></table></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
<script>
const urls=__URLS__; document.getElementById("urls").innerHTML=urls.map(([n,u])=>`<div>${n}: <code>${u}</code></div>`).join("");
if(window.QRCode&&urls.length) new QRCode(document.getElementById("qr"),{text:urls[0][1],width:96,height:96,correctLevel:QRCode.CorrectLevel.M});
if(window.QRCode&&urls.length) new QRCode(document.getElementById("qrca"),{text:urls[0][1].replace(/\/[^\/]+$/,"/ca.crt"),width:96,height:96,correctLevel:QRCode.CorrectLevel.M});
const c=(v,good,ok)=>v==null?"":(v<=good?"ok":v<=ok?"warn":"bad");
async function tick(){ const s=await (await fetch("/stats")).json(), p=s.phone||{}, L=s.latency_ms, G=s.gap_ms, A=s.audio_latency_ms;
 const rows=[["connected",s.connected?`yes (${s.transport})`:"no",s.connected?(s.transport==="websocket"?"ok":"warn"):"bad"],["stream",s.kind||"–"],["phone camera",p.camera||"–"],
  ["fps received",s.fps,s.fps>=25?"ok":s.fps>=20?"warn":"bad"],
  ["latency p50 / p90 / p99 / max (ms)",`${L.p50??"–"} / ${L.p90??"–"} / ${L.p99??"–"} / ${L.max??"–"}`,c(L.p90,50,80)],
  ["arrival gap p50 / p90 / max (ms)",`${G.p50??"–"} / ${G.p90??"–"} / ${G.max??"–"}`,c(G.max,60,120)],
  ["mic latency p50 / p90 (ms)",`${A.p50??"–"} / ${A.p90??"–"}`],["dropped frames",s.dropped],["bitrate",s.mbit_s+" Mbit/s"],
  ["laptop decode",(s.decode_ms??"–")+" ms"],["clock sync round trip",(s.clock_rtt_ms??"–")+" ms"],
  ["phone",p.error?`<span class="bad">${p.error}</span>`:(p.hello?`${p.hello.match(/OS [\d_]+/)?.[0]||""} · WebCodecs ${p.webcodecs?"yes":"no"} · rVFC ${p.rvfc?"yes":"no"}`:"–")],
  ["phone: encoded / sent fps",`${p.enc_fps??"–"} / ${p.sent_fps??"–"}`],["phone: encode time",(p.encode_ms??"–")+" ms"],["phone: socket backlog",(p.ws_buffer??"–")+" bytes"]];
 document.getElementById("t").innerHTML=rows.map(([k,v,cl])=>`<tr><td>${k}</td><td class="${cl||""}">${v}</td></tr>`).join(""); }
setInterval(tick,1000); tick();
</script></body></html>"""

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
