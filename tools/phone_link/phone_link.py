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
OUT = REPO / ".tmp" / "phone_link"
HDR = struct.Struct("<dIB")      # capture time (epoch ms, phone clock), seq, keyframe flag


def iface_ips():
    rows = [ln.split() for ln in subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True,
                                                 text=True).stdout.splitlines()]
    return {r[1]: r[3].split("/")[0] for r in rows if len(r) > 3 and not r[3].startswith("127.")
            and not r[1].startswith(("docker", "br-", "veth"))}


def cert(ips):
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
        self.csv = open(OUT / "frames.csv", "w")
        self.csv.write("t_arrival_ms,seq,key,bytes,latency_ms,decode_ms,rtt_ms\n")

    async def pinger(self, ws):
        while True:
            await ws.send(b"P" + struct.pack("<d", time.time() * 1000))
            await asyncio.sleep(0.5)

    def video(self, m, jpeg=False):
        t_arr = time.time() * 1000
        cap, seq, key = HDR.unpack_from(m, 1)
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
                if not isinstance(m, bytes) or not m:
                    continue
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
        except Exception:  # noqa: BLE001
            pass
        finally:
            pt.cancel()
            if self.ws is ws:
                self.ws = None
            print("[link] phone disconnected", flush=True)

    def stats(self):
        now = time.time()
        arr = [t for t in self.arr if now - t < 3]
        gaps = np.diff(arr) * 1000 if len(arr) > 2 else np.array([])
        lat = np.array(self.lat)
        alat = np.array(self.alat)
        br = sum(n for t, n in self.bytes if now - t < 3) * 8 / 3 / 1e6
        pct = lambda x, p: None if not len(x) else round(float(np.percentile(x, p)), 1)  # noqa: E731
        return {"connected": self.ws is not None, "kind": self.kind,
                "fps": round(len(arr) / 3, 1) if arr else 0.0,
                "latency_ms": {"p50": pct(lat, 50), "p90": pct(lat, 90), "p99": pct(lat, 99), "max": pct(lat, 100)},
                "gap_ms": {"p50": pct(gaps, 50), "p90": pct(gaps, 90), "max": pct(gaps, 100)},
                "audio_latency_ms": {"p50": pct(alat, 50), "p90": pct(alat, 90)},
                "decode_ms": pct(np.array(self.dec_ms), 50), "mbit_s": round(br, 2),
                "dropped": self.dropped, "frames": self.frames,
                "clock_rtt_ms": None if self.clock.rtt() is None else round(self.clock.rtt(), 1),
                "phone": self.phone if self.phone and now - self.phone.get("t", 0) < 5 else {}}

    def http(self, conn, req):
        if req.path.startswith("/ws"):
            return None
        if req.path.split("?")[0] == "/" and f"t={self.a.token}" in req.path:
            body = PHONE_PAGE.replace("__TOKEN__", self.a.token).replace("__BITRATE__", str(self.a.bitrate)).encode()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "text/html; charset=utf-8"),
                                                               ("Cache-Control", "no-store"),
                                                               ("Content-Length", str(len(body)))]), body)
        return Response(http.HTTPStatus.NOT_FOUND, "Not Found", Headers([("Content-Length", "0")]), b"")


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
    urls = [(name, f"https://{ip}:{a.port}/?t={a.token}") for name, ip in order]
    OUT.mkdir(parents=True, exist_ok=True)
    link = Link(a)
    laptop_server(link, a.view_port, urls)
    async with serve(link.handler, "0.0.0.0", a.port, ssl=cert([ip for _, ip in order]), process_request=link.http,
                     max_size=8 * 2**20, ping_interval=20):
        for name, u in urls:
            print(f"[link] phone URL via {name}: {u}", flush=True)
        print(f"[link] laptop view + numbers: http://127.0.0.1:{a.view_port}", flush=True)
        await asyncio.Future()


PHONE_PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Phone link test</title>
<style>[hidden]{display:none!important} body{margin:0;background:#000;color:#ddd;font:15px -apple-system,system-ui}
#me{width:100%;max-height:62vh;object-fit:contain;transform:scaleX(-1);background:#111}
#p{padding:12px} button{font:600 17px -apple-system;border:0;border-radius:22px;padding:12px 22px;background:#2e7d32;color:#fff}
pre{font:13px ui-monospace,monospace;white-space:pre-wrap}</style></head><body>
<video id="me" playsinline muted autoplay></video>
<div id="p"><button id="go">Start link test</button><pre id="st">Camera + mic go to the laptop only (local network).</pre></div>
<script>
const T="__TOKEN__", BR=__BITRATE__, $=id=>document.getElementById(id);
const now=()=>performance.timeOrigin+performance.now();
let ws, enc=null, seq=0, nEnc=0, nSent=0, encMs=[], kind="", fpsT=now();
function hdr(tag,cap,sq,key){const b=new ArrayBuffer(14); const v=new DataView(b); v.setUint8(0,tag); v.setFloat64(1,cap,true); v.setUint32(9,sq,true); v.setUint8(13,key?1:0); return new Uint8Array(b);}
function send(tag,cap,sq,key,data){ if(!ws||ws.readyState!==1)return; if(ws.bufferedAmount>2e6) return; // link can't keep up: drop
  const h=hdr(tag,cap,sq,key), o=new Uint8Array(h.length+data.byteLength); o.set(h,0); o.set(new Uint8Array(data),h.length); ws.send(o); nSent++; }
function connect(){ ws=new WebSocket(`wss://${location.host}/ws?t=${T}`); ws.binaryType="arraybuffer";
  ws.onmessage=e=>{const d=new DataView(e.data); if(d.getUint8(0)===80){ // 'P' ping: echo with our clock
      const o=new ArrayBuffer(17), v=new DataView(o); v.setUint8(0,81); v.setFloat64(1,d.getFloat64(1,true),true); v.setFloat64(9,now(),true); ws.send(o);} };
  ws.onclose=()=>setTimeout(connect,1000); }
$("go").onclick=async()=>{ $("go").hidden=true;
  const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user",width:{ideal:960},height:{ideal:720},frameRate:{ideal:30}},
     audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1}});
  const v=$("me"); v.srcObject=s; await v.play(); connect();
  const st=s.getVideoTracks()[0].getSettings(); let W=v.videoWidth&~1, H=v.videoHeight&~1;
  if("VideoEncoder" in window){
    kind="h264 (WebCodecs, hardware)";
    const cfg={codec:"avc1.42E01F",width:W,height:H,bitrate:BR,framerate:30,latencyMode:"realtime",hardwareAcceleration:"prefer-hardware",avc:{format:"annexb"}};
    let sup=await VideoEncoder.isConfigSupported(cfg).catch(()=>({supported:false}));
    if(!sup.supported){cfg.codec="avc1.640028"; sup=await VideoEncoder.isConfigSupported(cfg).catch(()=>({supported:false}));}
    if(sup.supported){
      const caps=new Map();
      enc=new VideoEncoder({output:(chunk)=>{const c=caps.get(chunk.timestamp)||{}; caps.delete(chunk.timestamp); encMs.push(now()-c.t0); if(encMs.length>60)encMs.shift();
          const b=new ArrayBuffer(chunk.byteLength); chunk.copyTo(b); send(72,c.cap,c.sq,chunk.type==="key",b); nEnc++;},
        error:e=>{$("st").textContent="encoder error: "+e.message;}});
      enc.configure(cfg);
      const step=(t,meta)=>{ if(enc.encodeQueueSize<2){ const cap=meta&&meta.captureTime?performance.timeOrigin+meta.captureTime:now();
          const ts=Math.round(meta.mediaTime*1e6)||Math.round(t*1000); const f=new VideoFrame(v,{timestamp:ts}); const sq=++seq;
          caps.set(ts,{cap,sq,t0:now()}); enc.encode(f,{keyFrame:sq%60===1}); f.close(); }
        v.requestVideoFrameCallback(step); };
      v.requestVideoFrameCallback(step);
    } else kind="";
  }
  if(!enc){ kind="jpeg fallback (no WebCodecs H.264)"; const cv=document.createElement("canvas"), g=cv.getContext("2d"); cv.width=W; cv.height=H;
    const step=(t,meta)=>{ const cap=meta&&meta.captureTime?performance.timeOrigin+meta.captureTime:now(); g.drawImage(v,0,0,W,H);
      const sq=++seq; cv.toBlob(async b=>{ if(b) send(74,cap,sq,true,await b.arrayBuffer()); nEnc++; },"image/jpeg",0.7); v.requestVideoFrameCallback(step); };
    v.requestVideoFrameCallback(step); }
  // mic: timestamped PCM (48 kHz s16)
  const ctx=new AudioContext(); await ctx.resume(); const src=ctx.createMediaStreamSource(s), pr=ctx.createScriptProcessor(1024,1,1), r=48000/ctx.sampleRate;
  pr.onaudioprocess=e=>{ if(!ws||ws.readyState!==1)return; const x=e.inputBuffer.getChannelData(0), n=Math.floor(x.length*r), o=new Int16Array(n);
    for(let i=0;i<n;i++)o[i]=Math.max(-1,Math.min(1,x[Math.min(x.length-1,Math.floor(i/r))]))*32767;
    const cap=now()-1000*x.length/ctx.sampleRate, b=new ArrayBuffer(9+o.byteLength), dv=new DataView(b); dv.setUint8(0,65); dv.setFloat64(1,cap,true); new Int16Array(b,9).set(o); ws.send(b);};
  const z=ctx.createGain(); z.gain.value=0; src.connect(pr); pr.connect(z); z.connect(ctx.destination);
  setInterval(()=>{ const dt=(now()-fpsT)/1000; fpsT=now(); const em=encMs.length?encMs.reduce((a,b)=>a+b)/encMs.length:0;
    const rep={kind, camera:`${st.width}x${st.height} @ ${st.frameRate||"?"} fps`, enc_fps:+(nEnc/dt).toFixed(1), sent_fps:+(nSent/dt).toFixed(1), encode_ms:+em.toFixed(1), ws_buffer:ws?ws.bufferedAmount:0};
    nEnc=0; nSent=0; $("st").textContent=Object.entries(rep).map(([k,v])=>`${k}: ${v}`).join("\n");
    if(ws&&ws.readyState===1) ws.send(new TextEncoder().encode("S"+JSON.stringify(rep))); },1000);
};
</script></body></html>"""

LAPTOP_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Phone Link</title>
<style>body{margin:0;background:#0e1013;color:#e6e8eb;font:14px system-ui;padding:16px} h1{font-size:16px;margin:0 0 10px}
.wrap{display:flex;gap:16px;flex-wrap:wrap} img{height:70vh;background:#000;border-radius:6px} table{border-collapse:collapse}
td{padding:3px 10px;border-bottom:1px solid #2a2f37;font-variant-numeric:tabular-nums} td:first-child{color:#8b939e}
.ok{color:#3fb27f}.warn{color:#e0a03a}.bad{color:#e05a4f} code{color:#5a9cf0;word-break:break-all}</style></head><body>
<h1>Phone link test</h1><div id="urls"></div>
<div class="wrap"><img id="v" src="/video"><table id="t"></table></div>
<script>
const urls=__URLS__; document.getElementById("urls").innerHTML=urls.map(([n,u])=>`<div>${n}: <code>${u}</code></div>`).join("");
const c=(v,good,ok)=>v==null?"":(v<=good?"ok":v<=ok?"warn":"bad");
async function tick(){ const s=await (await fetch("/stats")).json(), p=s.phone||{}, L=s.latency_ms, G=s.gap_ms, A=s.audio_latency_ms;
 const rows=[["connected",s.connected],["stream",s.kind||"–"],["phone camera",p.camera||"–"],
  ["fps received",s.fps,s.fps>=25?"ok":s.fps>=20?"warn":"bad"],
  ["latency p50 / p90 / p99 / max (ms)",`${L.p50??"–"} / ${L.p90??"–"} / ${L.p99??"–"} / ${L.max??"–"}`,c(L.p90,50,80)],
  ["arrival gap p50 / p90 / max (ms)",`${G.p50??"–"} / ${G.p90??"–"} / ${G.max??"–"}`,c(G.max,60,120)],
  ["mic latency p50 / p90 (ms)",`${A.p50??"–"} / ${A.p90??"–"}`],["dropped frames",s.dropped],["bitrate",s.mbit_s+" Mbit/s"],
  ["laptop decode",(s.decode_ms??"–")+" ms"],["clock sync round trip",(s.clock_rtt_ms??"–")+" ms"],
  ["phone: encoded / sent fps",`${p.enc_fps??"–"} / ${p.sent_fps??"–"}`],["phone: encode time",(p.encode_ms??"–")+" ms"],["phone: socket backlog",(p.ws_buffer??"–")+" bytes"]];
 document.getElementById("t").innerHTML=rows.map(([k,v,cl])=>`<tr><td>${k}</td><td class="${cl||""}">${v}</td></tr>`).join(""); }
setInterval(tick,1000); tick();
</script></body></html>"""

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
