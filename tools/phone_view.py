"""Phone client for the live character mode: watch + hear the character and talk to it from a phone on the same Wi-Fi.

  laptop -> phone : the output video (dashboard /stream/final, JPEG) + the character's voice (PulseAudio sink OUT_SINK,
                    the bridge's audio monitor plays into it), over one WebSocket, re-synced on the phone
  phone  -> laptop: the phone's mic (echo-cancelled by the browser) -> PulseAudio sink MIC_SINK; its .monitor is the
                    "mic" the driver listens to (nuance_source --mic MIC_SINK.monitor)
                    + the phone's front camera (JPEG, ~15 fps) -> raw RGBA 480x640 on tcp://127.0.0.1:CAMERA_PORT
                    (nuance_source --camera tcp://127.0.0.1:CAMERA_PORT: the persona sees you)

HTTPS with a self-signed certificate (browsers only allow the mic on secure pages): the phone shows a warning once
(Safari: Show Details -> visit this website). A random token in the link keeps others on the Wi-Fi out.
Started by tools/character_live.py --phone; standalone:
  cloud/facefix/.venv/bin/python tools/phone_view.py --out-sink phone_out --mic-sink phone_mic [--port 8443]
"""
import argparse
import asyncio
import http
import json
import secrets
import socket
import ssl
import subprocess
import threading
import time
import collections
import urllib.request
from pathlib import Path

import struct

import cv2
import numpy as np

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

REPO = Path(__file__).resolve().parents[1]
SR, CHUNK = 48000, 960  # 20 ms audio chunks
CAM_W, CAM_H = 480, 640  # camera frames handed on (nuance_source CAM_W/CAM_H)
DRV_W, DRV_H = 720, 960  # --drive-srt: the phone camera as the pipeline's source video (3:4 portrait)


def lan_ip(iface=None):
    """The address the phone should open. Order: --iface; the Beryl LAN (192.168.8.0/24: the Seattle phone's only
    network -- it must never be offered another); Wi-Fi (wl*); Ethernet. (A VPN may own the default route, so ask the
    interfaces.)"""
    rows = [ln.split() for ln in subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True,
                                                 text=True).stdout.splitlines()]
    addrs = [(r[1], r[3].split("/")[0]) for r in rows if len(r) > 3]
    for name, ip in addrs:
        if iface and name == iface:
            return ip
    for name, ip in addrs:
        if ip.startswith("192.168.8."):
            return ip
    for pref in ("wl", "en", "eth"):
        for name, ip in addrs:
            if name.startswith(pref) and not ip.startswith("127."):
                return ip
    return "127.0.0.1"


def cert(ip):
    """Signed by the local CA (tools/local_ca.py; the phone trusts it once): no warning, and Safari's WebSockets
    connect (with a self-signed certificate they can hang)."""
    import sys as _s
    _s.path.insert(0, str(REPO / "tools"))
    from local_ca import server_cert
    crt, key = server_cert([ip], REPO / ".tmp" / "phone_cert")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    return ctx


def cert_selfsigned(ip):
    d = REPO / ".tmp" / "phone_cert"
    d.mkdir(parents=True, exist_ok=True)
    crt, key, tag = d / "cert.pem", d / "key.pem", d / "ip.txt"
    if not crt.exists() or not tag.exists() or tag.read_text() != ip:
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825", "-keyout", str(key),
                        "-out", str(crt), "-subj", "/CN=ai-livestream-phone", "-addext",
                        f"subjectAltName=IP:{ip},DNS:localhost"], check=True, capture_output=True)
        tag.write_text(ip)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    return ctx


class Clock:
    """Phone clock -> laptop clock: offset from the ping with the smallest round trip (last ~30 s)."""

    def __init__(self):
        self.samples = collections.deque(maxlen=60)

    def add(self, t_send, t_phone, t_recv):
        rtt = t_recv - t_send
        self.samples.append((rtt, t_phone - (t_send + rtt / 2)))

    def offset(self):
        return min(self.samples)[1] if self.samples else None


DRV_HDR = struct.Struct("<dIB")    # phone page 'H': capture time (phone clock, ms), seq, flags (bit0 key, bits1-2 rot)
FWD_HDR = struct.Struct("<IdB")    # to the bridge: length, capture time (laptop clock, ms), flags


class Hub:
    def __init__(self, a):
        self.a, self.clients, self.loop = a, set(), None
        self.cam_out = set()   # local camera readers (asyncio StreamWriters)
        self.enc = None        # --drive-srt encoder (ffmpeg -> SRT -> the bridge)
        # counters for the console (GET /stats): per second, over the last ~2 s
        self.cnt = {k: 0 for k in ("cam_in", "mic_in", "video_out", "audio_out")}
        self.rates, self.t_rate = {}, time.time()
        self.mic_rms = collections.deque(maxlen=25)
        self.cam_gaps, self.t_cam = collections.deque(maxlen=150), None   # camera frame arrival gaps (burstiness)
        self.clock = Clock()
        self.h264_out = set()       # --drive-h264-port readers (the bridge)
        self.link_ms = collections.deque(maxlen=150)   # phone capture -> here (ms)
        self.want_key = False
        self.phone = {}        # what the page reports (audio buffer, frames shown, output latency, sync offset)
        self.n_cam = 0
        self.mic = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                     "--latency-msec=20", "-d", a.mic_sink], stdin=subprocess.PIPE)

    def tick(self):
        now = time.time()
        if now - self.t_rate >= 2:
            dt = now - self.t_rate
            self.rates = {k: round(v / dt, 1) for k, v in self.cnt.items()}
            self.cnt = {k: 0 for k in self.cnt}
            self.t_rate = now

    def stats(self):
        self.tick()
        m = max(self.mic_rms) if self.mic_rms else 0
        g = sorted(self.cam_gaps)
        burst = {"gap_p50_ms": round(g[len(g) // 2] * 1000), "gap_p90_ms": round(g[int(len(g) * 0.9)] * 1000),
                 "gap_max_ms": round(g[-1] * 1000)} if len(g) > 20 else None
        lk = sorted(self.link_ms)
        link = {"p50": round(lk[len(lk) // 2]), "p90": round(lk[int(len(lk) * 0.9)]), "max": round(lk[-1])} if len(lk) > 20 else None
        return {"clients": len(self.clients), "rates": self.rates, "cam_arrival": burst, "link_ms": link,
                "mic_db": round(20 * np.log10(max(1e-5, m)), 1), "phone": self.phone, "drive": bool(self.a.drive_srt)}

    def broadcast(self, data, video=False):
        self.cnt["video_out" if video else "audio_out"] += 1
        for ws in list(self.clients):
            tr = ws.transport
            if video and tr is not None and tr.get_write_buffer_size() > 600_000:
                continue   # slow phone link: skip this frame rather than build a backlog
            asyncio.run_coroutine_threadsafe(self._send(ws, data), self.loop)

    @staticmethod
    async def _send(ws, data):
        try:
            await ws.send(data)
        except Exception:  # noqa: BLE001
            pass

    def video_thread(self):
        url = f"http://127.0.0.1:{self.a.dashboard}/stream/{self.a.view}?fps={self.a.fps}"
        while True:
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    buf = b""
                    while True:
                        buf += r.read1(65536)
                        while True:
                            i = buf.find(b"Content-Length: ")
                            if i < 0:
                                break
                            j = buf.find(b"\r\n\r\n", i)
                            if j < 0:
                                break
                            n = int(buf[i + 16:buf.find(b"\r\n", i)])
                            if len(buf) < j + 4 + n:
                                break
                            jpg = buf[j + 4:j + 4 + n]
                            buf = buf[j + 4 + n:]
                            if self.clients:
                                self.broadcast(b"V" + jpg, video=True)
            except Exception:  # noqa: BLE001  (bridge not up yet / restarted)
                time.sleep(1)

    def audio_thread(self):
        while True:
            p = subprocess.Popen(["parec", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                  "--latency-msec=20", "-d", f"{self.a.out_sink}.monitor"], stdout=subprocess.PIPE)
            while True:
                b = p.stdout.read(CHUNK * 2)
                if len(b) < CHUNK * 2:
                    break
                if self.clients:
                    self.broadcast(b"A" + b)
            p.kill()
            time.sleep(1)

    async def handler(self, ws):
        if not ws.request.path.startswith(f"/ws?t={self.a.token}"):
            await ws.close(1008, "bad token")
            return
        for old in list(self.clients):   # one phone at a time: a reload / second tab replaces the old one
            self.clients.discard(old)
            asyncio.ensure_future(old.close(1000, "replaced by a newer connection"))
        self.clients.add(ws)
        ping_task = asyncio.create_task(self.pinger(ws))
        self.want_key = True
        print(f"[phone] connected ({len(self.clients)})", flush=True)
        try:
            async for m in ws:
                if not isinstance(m, bytes) or not m:
                    continue
                if m[:1] == b"M" and self.mic.poll() is None:
                    self.cnt["mic_in"] += 1
                    x = np.frombuffer(m[1:], np.int16).astype(np.float32) / 32768
                    self.mic_rms.append(float(np.sqrt((x ** 2).mean())) if len(x) else 0.0)
                    self.mic.stdin.write(m[1:])
                    self.mic.stdin.flush()
                elif m[:1] == b"S":
                    try:
                        self.phone = {**json.loads(m[1:].decode()), "t": time.time()}
                    except ValueError:
                        pass
                elif m[:1] == b"Q":
                    t_send, t_phone = struct.unpack_from("<dd", m, 1)
                    self.clock.add(t_send, t_phone, time.time() * 1000)
                elif m[:1] == b"H":
                    self.h264(m)
                    if self.want_key:
                        self.want_key = False
                        await ws.send(b"K")
                elif m[:1] == b"C" and (self.cam_out or self.a.drive_srt):
                    self.camera_frame(m[1:])
        finally:
            ping_task.cancel()
            self.clients.discard(ws)
            print(f"[phone] disconnected ({len(self.clients)})", flush=True)

    def h264(self, m):
        """The phone's hardware H.264 (WebCodecs), passed to the bridge with its capture time on the laptop clock."""
        now = time.time() * 1000
        cap, _seq, flags = DRV_HDR.unpack_from(m, 1)
        data = m[1 + DRV_HDR.size:]
        off = self.clock.offset()
        cap_l = now if off is None else cap - off
        if off is not None:
            self.link_ms.append(now - cap_l)
        self.cnt["cam_in"] += 1
        if self.t_cam:
            self.cam_gaps.append(now / 1000 - self.t_cam)
        self.t_cam = now / 1000
        pkt = FWD_HDR.pack(len(data), cap_l, flags) + data
        for wr in list(self.h264_out):
            if wr.transport.get_write_buffer_size() > 2_000_000:
                continue
            wr.write(pkt)
        self.n_cam += 1
        if self.n_cam == 1:
            print("[phone] camera (H.264) -> the pipeline", flush=True)

    async def h264_reader(self, r, w):
        self.h264_out.add(w)
        self.want_key = True        # a new decoder needs a keyframe: ask the phone for one now
        print("[phone] H.264 reader connected (the bridge)", flush=True)
        try:
            await r.read()
        except (ConnectionError, OSError):
            pass
        finally:
            self.h264_out.discard(w)

    async def pinger(self, ws):
        while True:
            try:
                await ws.send(b"P" + struct.pack("<d", time.time() * 1000))
            except Exception:  # noqa: BLE001
                return
            await asyncio.sleep(0.5)

    def drive(self, img):
        """The phone camera as the source video: raw frames -> x264 -> SRT (the bridge listens)."""
        if self.enc is None or self.enc.poll() is not None:
            self.enc = subprocess.Popen(
                ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{DRV_W}x{DRV_H}",
                 "-use_wallclock_as_timestamps", "1", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                 "-tune", "zerolatency", "-g", "30", "-b:v", "4M", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
                 "-f", "mpegts", self.a.drive_srt], stdin=subprocess.PIPE)
            print(f"[phone] camera -> {self.a.drive_srt} (the source video)", flush=True)
        try:
            self.enc.stdin.write(img.tobytes())
        except (BrokenPipeError, OSError):
            self.enc = None

    def camera_frame(self, jpg):
        self.cnt["cam_in"] += 1
        now = time.time()
        if self.t_cam:
            self.cam_gaps.append(now - self.t_cam)
        self.t_cam = now
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        if self.a.drive_srt:
            h, w = img.shape[:2]
            if w * DRV_H > h * DRV_W:
                nw = h * DRV_W // DRV_H
                c = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
            else:
                nh = w * DRV_H // DRV_W
                c = img[(h - nh) // 2:(h - nh) // 2 + nh]
            self.drive(cv2.resize(c, (DRV_W, DRV_H), interpolation=cv2.INTER_AREA))
            self.n_cam += 1
            if self.n_cam == 1:
                print("[phone] camera: first frame -> the pipeline", flush=True)
            if not self.cam_out:
                return
        h, w = img.shape[:2]
        if w * CAM_H > h * CAM_W:            # crop to 3:4 portrait, then scale
            nw = h * CAM_W // CAM_H
            img = img[:, (w - nw) // 2:(w - nw) // 2 + nw]
        else:
            nh = w * CAM_H // CAM_W
            img = img[(h - nh) // 2:(h - nh) // 2 + nh]
        rgba = cv2.cvtColor(cv2.resize(img, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGBA)
        data = rgba.tobytes()
        for wr in list(self.cam_out):
            if wr.transport.get_write_buffer_size() > 4 * len(data):
                continue   # reader behind: drop
            wr.write(data)
        self.n_cam += 1
        if self.n_cam == 1:
            print("[phone] camera: first frame handed on", flush=True)

    async def camera_reader(self, r, w):
        self.cam_out.add(w)
        print("[phone] camera reader connected", flush=True)
        try:
            await r.read()     # until it disconnects
        except (ConnectionError, OSError):
            pass
        finally:
            self.cam_out.discard(w)

    def http(self, conn, req):
        if req.path.startswith("/ws"):
            return None  # WebSocket upgrade
        if req.path.split("?")[0] == "/ca.crt":   # the local CA, for the phone to install + trust once
            from local_ca import CA_DER, ensure_ca
            ensure_ca()
            body = CA_DER.read_bytes()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "application/x-x509-ca-cert"),
                                                               ("Content-Disposition", 'attachment; filename="ai-livestream-local-ca.crt"'),
                                                               ("Content-Length", str(len(body)))]), body)

        if req.path.split("?")[0] == "/stats" and f"t={self.a.token}" in req.path:
            body = json.dumps(self.stats()).encode()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "application/json"),
                                                               ("Content-Length", str(len(body)))]), body)
        if req.path.split("?")[0] == "/" and f"t={self.a.token}" in req.path:
            body = (PAGE.replace("__TOKEN__", self.a.token).replace("__TITLE__", self.a.title)
                    .replace("__CAMERA__", "true" if self.a.camera_port or self.a.drive_srt or self.a.drive_h264_port
                             else "false")
                    .replace("__DRIVE__", "true" if self.a.drive_srt or self.a.drive_h264_port else "false")
                    .replace("__H264__", "true" if self.a.drive_h264_port else "false")).encode()
            return Response(http.HTTPStatus.OK, "OK", Headers([("Content-Type", "text/html; charset=utf-8"),
                                                               ("Cache-Control", "no-store"),
                                                               ("Content-Length", str(len(body)))]), body)
        return Response(http.HTTPStatus.NOT_FOUND, "Not Found", Headers([("Content-Length", "0")]), b"")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-sink", default="phone_out")
    ap.add_argument("--mic-sink", default="phone_mic")
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--dashboard", type=int, default=8765)
    ap.add_argument("--view", default="final")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--token", default=None)
    ap.add_argument("--title", default="Character")
    ap.add_argument("--drive-srt", default=None, help="the phone camera becomes the source video: H.264 over SRT to "
                    "this URL (caller), e.g. srt://127.0.0.1:9024 (the bridge's --input srt://0.0.0.0:9024)")
    ap.add_argument("--iface", default=None, help="offer the phone this interface's address (default: the Beryl LAN "
                    "if present, else Wi-Fi)")
    ap.add_argument("--drive-h264-port", type=int, default=0, help="the phone camera becomes the source video as the "
                    "phone's own H.264 (WebCodecs) with capture times, served to the bridge on tcp://127.0.0.1:PORT "
                    "(bridge --input h264tcp://127.0.0.1:PORT)")
    ap.add_argument("--camera-port", type=int, default=0, help="hand the phone camera on to tcp://127.0.0.1:PORT "
                    "(0 = the page doesn't ask for the camera)")
    a = ap.parse_args()
    a.token = a.token or secrets.token_urlsafe(9)
    ip = lan_ip(a.iface)
    hub = Hub(a)
    hub.loop = asyncio.get_running_loop()
    threading.Thread(target=hub.video_thread, daemon=True).start()
    threading.Thread(target=hub.audio_thread, daemon=True).start()
    if a.camera_port:
        await asyncio.start_server(hub.camera_reader, "127.0.0.1", a.camera_port)
    if a.drive_h264_port:
        await asyncio.start_server(hub.h264_reader, "127.0.0.1", a.drive_h264_port)
    async with serve(hub.handler, "0.0.0.0", a.port, ssl=cert(ip), process_request=hub.http, max_size=2**20,
                     ping_interval=20):
        print(f"[phone] open on the phone (same Wi-Fi): https://{ip}:{a.port}/?t={a.token}", flush=True)
        await asyncio.Future()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>__TITLE__</title>
<style>
[hidden]{display:none!important}
html,body{margin:0;height:100%;background:#000;color:#ddd;font:15px -apple-system,system-ui,sans-serif}
#v{position:fixed;inset:0;width:100%;height:100%;object-fit:contain}
#ui{position:fixed;left:0;right:0;bottom:0;padding:12px max(12px,env(safe-area-inset-left)) calc(12px + env(safe-area-inset-bottom));
 display:flex;gap:10px;align-items:center;background:linear-gradient(transparent,#000c)}
button{font:600 16px -apple-system,system-ui;border:0;border-radius:22px;padding:11px 18px;background:#2e7d32;color:#fff}
button.off{background:#8a3b2b} #st{font-size:12px;opacity:.8;flex:1;text-align:right}
#sync{display:flex;align-items:center;gap:6px;font-size:12px;opacity:.85} #sync input{width:110px}
#go{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;background:#000}
#go p{max-width:300px;text-align:center;opacity:.75;margin:0}
#me{position:fixed;top:calc(10px + env(safe-area-inset-top));right:max(10px,env(safe-area-inset-right));width:24vw;
 max-width:130px;border-radius:10px;transform:scaleX(-1);border:1px solid #fff4;background:#111}
</style></head><body>
<img id="v" alt=""><video id="me" playsinline muted autoplay hidden></video>
<div id="go"><button id="start">Start talking to __TITLE__</button><p>Allow the microphone (and camera: __TITLE__ sees you). Earbuds work best (the phone
cancels echo, but speakers can still leak into the mic).</p></div>
<div id="ui" hidden><button id="mute">Mic on</button>
<label id="sync" title="Voice comes late (e.g. Bluetooth earbuds): slide right. Voice early: slide left.">sync
<input id="off" type="range" min="-400" max="400" step="20" value="0"><span id="offv">0</span></label><span id="st"></span></div>
<script>
const T="__TOKEN__", CAM=__CAMERA__, DRIVE=__DRIVE__, H264=__H264__, $=id=>document.getElementById(id);
const nowMs=()=>performance.timeOrigin+performance.now(); let forceKey=false;
let ctx, ws, micOn=true, playAt=0, frames=[], lastURL=null, rx=0, lag=0.15, userOff=0;
try{userOff=+localStorage.getItem("avoff")||0;}catch(e){}
// the picture is shown when its audio is heard: audio buffered + the device's audio output latency (+ user's offset,
// e.g. Bluetooth earbuds); the sound is late if > 0, so the picture waits; < 0 = the voice waits instead
function outLat(){return (ctx.outputLatency||0)+(ctx.baseLatency||0);}
function connect(){
  ws=new WebSocket(`wss://${location.host}/ws?t=${T}`); ws.binaryType="arraybuffer";
  ws.onmessage=e=>{const b=new Uint8Array(e.data), k=b[0];
    if(k===80){ // 'P' clock ping: echo with our clock (the laptop maps capture times onto its own)
      const o=new ArrayBuffer(17), v=new DataView(o); v.setUint8(0,81); v.setFloat64(1,new DataView(e.data).getFloat64(1,true),true); v.setFloat64(9,nowMs(),true); ws.send(o); return;}
    if(k===75){ forceKey=true; return; }   // 'K': the pipeline (re)connected and needs a keyframe
    if(k===65){ // 'A' audio: s16le 48 kHz mono, scheduled with a small jitter buffer
      const s=new Int16Array(e.data.slice(1)), buf=ctx.createBuffer(1,s.length,48000), d=buf.getChannelData(0);
      for(let i=0;i<s.length;i++)d[i]=s[i]/32768;
      const now=ctx.currentTime, l=lag+Math.max(0,-userOff/1000); if(playAt<now+0.04||playAt>now+0.6+l)playAt=now+l;
      const n=ctx.createBufferSource(); n.buffer=buf; n.connect(ctx.destination); n.start(playAt); playAt+=buf.duration;
    }else if(k===86){ // 'V' JPEG: shown as late as the audio is buffered, so picture and voice stay together
      frames.push([performance.now()+Math.max(0,(playAt-ctx.currentTime)+outLat())*1000+Math.max(0,userOff), new Blob([b.subarray(1)],{type:"image/jpeg"})]);
      if(frames.length>40)frames.shift();
    } rx+=b.length;};
  ws.onclose=e=>{ if(e.reason&&e.reason.startsWith("replaced")){$("st").textContent="opened elsewhere"; return;}
    $("st").textContent="reconnecting…"; setTimeout(connect,1000);};
  ws.onopen=()=>{$("st").textContent="connected";};
}
function show(){const now=performance.now(); let f=null;
  while(frames.length&&frames[0][0]<=now)f=frames.shift();
  if(f){shownN++; const u=URL.createObjectURL(f[1]); $("v").src=u; if(lastURL)URL.revokeObjectURL(lastURL); lastURL=u;}
  requestAnimationFrame(show);}
let shownN=0;
setInterval(()=>{ if(!ws||ws.readyState!==1||!ctx)return;
  const st={buf_ms:Math.round(Math.max(0,playAt-ctx.currentTime)*1000), out_lat_ms:Math.round(outLat()*1000),
            shown_fps:+(shownN/2).toFixed(1), queued:frames.length, offset_ms:userOff, sr:ctx.sampleRate};
  shownN=0; ws.send(new TextEncoder().encode("S"+JSON.stringify(st)));},2000);
setInterval(()=>{if(ws&&ws.readyState===1)$("st").textContent=`${(rx*8/1e6/2).toFixed(1)} Mbit/s · audio buffer ${Math.round(Math.max(0,playAt-ctx.currentTime)*1000)} ms`; rx=0;},2000);
$("start").onclick=async()=>{
  if($("start").disabled)return; $("start").disabled=true; $("start").textContent="Starting…";
  ctx=new (window.AudioContext||window.webkitAudioContext)(); await ctx.resume();
  let stream;
  const ac={echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1};
  try{stream=await navigator.mediaDevices.getUserMedia(CAM?{audio:ac,video:DRIVE?{facingMode:"user",width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}}:{facingMode:"user",width:{ideal:640},height:{ideal:480}}}:{audio:ac});}
  catch(e){alert("Microphone/camera blocked: "+e.message); $("start").disabled=false; $("start").textContent="Start talking to __TITLE__"; return;}
  if(CAM&&stream.getVideoTracks().length&&H264&&!("VideoEncoder" in window)){ alert("This browser has no WebCodecs video encoder (needed to drive the character)."); }
  if(CAM&&stream.getVideoTracks().length&&H264&&("VideoEncoder" in window)){
    // your camera IS the source video: the phone's hardware H.264 encoder on the raw (sensor-oriented) frames, each
    // stamped with its capture time; the laptop rotates upright and measures the latency from capture
    const me=$("me"); me.srcObject=stream; me.hidden=false; await me.play().catch(()=>{});
    if(!me.videoWidth) await new Promise(r=>me.onloadedmetadata=r);
    const probe=new VideoFrame(me,{timestamp:0}); const FW=probe.displayWidth&~1, FH=probe.displayHeight&~1; probe.close();
    const ROT=((FW>FH)!==(me.videoWidth>me.videoHeight))?1:0, q=[]; let sq=0;
    const enc=new VideoEncoder({output:ch=>{const c=q.shift()||{}; const b=new Uint8Array(14+ch.byteLength), v=new DataView(b.buffer);
        v.setUint8(0,72); v.setFloat64(1,c.cap||nowMs(),true); v.setUint32(9,c.sq||0,true); v.setUint8(13,(ch.type==="key"?1:0)|(ROT<<1));
        ch.copyTo(b.subarray(14)); if(ws&&ws.readyState===1&&ws.bufferedAmount<2e6) ws.send(b);}, error:e=>alert("encoder: "+e.message)});
    enc.configure({codec:"avc1.42E01F",width:FW,height:FH,bitrate:4000000,framerate:30,latencyMode:"realtime",hardwareAcceleration:"prefer-hardware",avc:{format:"annexb"}});
    const step=(t,meta)=>{ if(enc.encodeQueueSize<2){ const cap=meta&&meta.captureTime?performance.timeOrigin+meta.captureTime:nowMs();
        const f=new VideoFrame(me,{timestamp:Math.round((meta&&meta.mediaTime||t/1000)*1e6)}); sq++; q.push({cap,sq});
        enc.encode(f,{keyFrame:forceKey||sq%60===1}); forceKey=false; f.close(); }
      me.requestVideoFrameCallback(step); };
    me.requestVideoFrameCallback(step);
  } else if(CAM&&stream.getVideoTracks().length){ // JPEG frames: ~15 fps for the persona to see you (or the
                                          // older JPEG drive path)
    const me=$("me"); me.srcObject=stream; me.hidden=false; await me.play().catch(()=>{});
    const cv=document.createElement("canvas"), g=cv.getContext("2d"); let busy=false;
    setInterval(()=>{ if(busy||!ws||ws.readyState!==1||!me.videoWidth||ws.bufferedAmount>300000)return;
      const s=Math.min(1,(DRIVE?960:640)/Math.max(me.videoWidth,me.videoHeight)); cv.width=me.videoWidth*s|0; cv.height=me.videoHeight*s|0;
      g.drawImage(me,0,0,cv.width,cv.height); busy=true;
      cv.toBlob(async b=>{busy=false; if(!b||!ws||ws.readyState!==1)return; const d=new Uint8Array(await b.arrayBuffer()), o=new Uint8Array(d.length+1); o[0]=67; o.set(d,1); ws.send(o);},"image/jpeg",DRIVE?0.82:0.7);
    },DRIVE?33:66);
  }
  const src=ctx.createMediaStreamSource(stream), proc=ctx.createScriptProcessor(1024,1,1), r=48000/ctx.sampleRate;
  proc.onaudioprocess=e=>{ if(!ws||ws.readyState!==1)return;
    const x=e.inputBuffer.getChannelData(0), n=Math.floor(x.length*r), o=new Int16Array(n);
    for(let i=0;i<n;i++){const v=micOn?x[Math.min(x.length-1,Math.floor(i/r))]:0; o[i]=Math.max(-1,Math.min(1,v))*32767;}
    const m=new Uint8Array(1+o.byteLength); m[0]=77; m.set(new Uint8Array(o.buffer),1); ws.send(m);};
  const z=ctx.createGain(); z.gain.value=0; src.connect(proc); proc.connect(z); z.connect(ctx.destination);
  $("go").hidden=true; $("ui").hidden=false; connect(); requestAnimationFrame(show);
  try{await navigator.wakeLock.request("screen");}catch(e){}
};
$("off").value=userOff; $("offv").textContent=userOff;
$("off").oninput=e=>{userOff=+e.target.value; $("offv").textContent=userOff; try{localStorage.setItem("avoff",userOff);}catch(_){}};
$("mute").onclick=()=>{micOn=!micOn; $("mute").textContent=micOn?"Mic on":"Mic muted"; $("mute").className=micOn?"":"off";};
</script></body></html>"""

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
