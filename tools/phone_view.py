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
import secrets
import socket
import ssl
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

REPO = Path(__file__).resolve().parents[1]
SR, CHUNK = 48000, 960  # 20 ms audio chunks
CAM_W, CAM_H = 480, 640  # camera frames handed on (nuance_source CAM_W/CAM_H)


def lan_ip():
    """The Wi-Fi address (a VPN such as Mullvad owns the default route, so ask the interfaces, wl* first)."""
    rows = [ln.split() for ln in subprocess.run(["ip", "-o", "-4", "addr", "show"], capture_output=True,
                                                 text=True).stdout.splitlines()]
    addrs = [(r[1], r[3].split("/")[0]) for r in rows if len(r) > 3]
    for pref in ("wl", "en", "eth"):
        for name, ip in addrs:
            if name.startswith(pref) and not ip.startswith("127."):
                return ip
    return "127.0.0.1"


def cert(ip):
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


class Hub:
    def __init__(self, a):
        self.a, self.clients, self.loop = a, set(), None
        self.cam_out = set()   # local camera readers (asyncio StreamWriters)
        self.n_cam = 0
        self.mic = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                     "--latency-msec=20", "-d", a.mic_sink], stdin=subprocess.PIPE)

    def broadcast(self, data, video=False):
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
        print(f"[phone] connected ({len(self.clients)})", flush=True)
        try:
            async for m in ws:
                if not isinstance(m, bytes) or not m:
                    continue
                if m[:1] == b"M" and self.mic.poll() is None:
                    self.mic.stdin.write(m[1:])
                    self.mic.stdin.flush()
                elif m[:1] == b"C" and self.cam_out:
                    self.camera_frame(m[1:])
        finally:
            self.clients.discard(ws)
            print(f"[phone] disconnected ({len(self.clients)})", flush=True)

    def camera_frame(self, jpg):
        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
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
        if req.path.split("?")[0] == "/" and f"t={self.a.token}" in req.path:
            body = (PAGE.replace("__TOKEN__", self.a.token).replace("__TITLE__", self.a.title)
                    .replace("__CAMERA__", "true" if self.a.camera_port else "false")).encode()
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
    ap.add_argument("--camera-port", type=int, default=0, help="hand the phone camera on to tcp://127.0.0.1:PORT "
                    "(0 = the page doesn't ask for the camera)")
    a = ap.parse_args()
    a.token = a.token or secrets.token_urlsafe(9)
    ip = lan_ip()
    hub = Hub(a)
    hub.loop = asyncio.get_running_loop()
    threading.Thread(target=hub.video_thread, daemon=True).start()
    threading.Thread(target=hub.audio_thread, daemon=True).start()
    if a.camera_port:
        await asyncio.start_server(hub.camera_reader, "127.0.0.1", a.camera_port)
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
const T="__TOKEN__", CAM=__CAMERA__, $=id=>document.getElementById(id);
let ctx, ws, micOn=true, playAt=0, frames=[], lastURL=null, rx=0, lag=0.15, userOff=0;
try{userOff=+localStorage.getItem("avoff")||0;}catch(e){}
// the picture is shown when its audio is heard: audio buffered + the device's audio output latency (+ user's offset,
// e.g. Bluetooth earbuds); the sound is late if > 0, so the picture waits; < 0 = the voice waits instead
function outLat(){return (ctx.outputLatency||0)+(ctx.baseLatency||0);}
function connect(){
  ws=new WebSocket(`wss://${location.host}/ws?t=${T}`); ws.binaryType="arraybuffer";
  ws.onmessage=e=>{const b=new Uint8Array(e.data), k=b[0];
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
  if(f){const u=URL.createObjectURL(f[1]); $("v").src=u; if(lastURL)URL.revokeObjectURL(lastURL); lastURL=u;}
  requestAnimationFrame(show);}
setInterval(()=>{if(ws&&ws.readyState===1)$("st").textContent=`${(rx*8/1e6/2).toFixed(1)} Mbit/s · audio buffer ${Math.round(Math.max(0,playAt-ctx.currentTime)*1000)} ms`; rx=0;},2000);
$("start").onclick=async()=>{
  if($("start").disabled)return; $("start").disabled=true; $("start").textContent="Starting…";
  ctx=new (window.AudioContext||window.webkitAudioContext)(); await ctx.resume();
  let stream;
  const ac={echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1};
  try{stream=await navigator.mediaDevices.getUserMedia(CAM?{audio:ac,video:{facingMode:"user",width:{ideal:640},height:{ideal:480}}}:{audio:ac});}
  catch(e){alert("Microphone/camera blocked: "+e.message); $("start").disabled=false; $("start").textContent="Start talking to __TITLE__"; return;}
  if(CAM&&stream.getVideoTracks().length){ // the persona sees you: JPEG frames at ~15 fps
    const me=$("me"); me.srcObject=stream; me.hidden=false; await me.play().catch(()=>{});
    const cv=document.createElement("canvas"), g=cv.getContext("2d"); let busy=false;
    setInterval(()=>{ if(busy||!ws||ws.readyState!==1||!me.videoWidth||ws.bufferedAmount>300000)return;
      const s=Math.min(1,640/Math.max(me.videoWidth,me.videoHeight)); cv.width=me.videoWidth*s|0; cv.height=me.videoHeight*s|0;
      g.drawImage(me,0,0,cv.width,cv.height); busy=true;
      cv.toBlob(async b=>{busy=false; if(!b||!ws||ws.readyState!==1)return; const d=new Uint8Array(await b.arrayBuffer()), o=new Uint8Array(d.length+1); o[0]=67; o.set(d,1); ws.send(o);},"image/jpeg",0.7);
    },66);
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
