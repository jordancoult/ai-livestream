"""Watch-only monitor for a second phone (e.g. the iPhone 17 Pro on the laptop's home Wi-Fi): the final output video
+ the character's voice, in sync. Isolated from the pipeline: its own process (low priority), it only TAPS what
exists (the bridge dashboard's final view, the voice's PulseAudio sink monitor), per-viewer bounded queues (a slow
viewer drops its own frames), and it listens only on the address you give it (default: the laptop's Wi-Fi). Survives
pipeline restarts ("waiting for the stream" in between). Transport: HTTPS polling like the phone page (Safari's
WebSockets hang at times); certificates from the local CA (tools/local_ca.py) -- or accept the warning once.

  cloud/facefix/.venv/bin/python tools/monitor_view.py [--iface wlo1] [--port 8450]
  -> prints the monitor link (stable token in .tmp/monitor_token); tools/character_console.py shows it too
"""
import argparse
import asyncio
import collections
import json
import os
import secrets
import ssl
import struct
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from local_ca import server_cert  # noqa: E402

SR, CHUNK = 48000, 960


def iface_ip(name):
    for ln in subprocess.run(["ip", "-o", "-4", "addr", "show", "dev", name], capture_output=True,
                             text=True).stdout.splitlines():
        return ln.split()[3].split("/")[0]
    return None


class Monitor:
    def __init__(self, a):
        self.a, self.viewers, self.loop = a, {}, None        # viewer id -> {"q": deque, "t": last poll}
        self.live = 0.0

    def push(self, msg, video):
        now = time.time()
        for vid, v in list(self.viewers.items()):
            if now - v["t"] > 10:
                del self.viewers[vid]
                continue
            q = v["q"]
            if video:
                if sum(1 for x in q if x[:1] == b"V") >= 2:     # keep only the newest frames waiting
                    for x in list(q):
                        if x[:1] == b"V":
                            q.remove(x)
                            break
            elif len(q) > 150:
                continue
            q.append(msg)

    def video_thread(self):
        url = f"http://127.0.0.1:{self.a.dashboard}/stream/final?fps={self.a.fps}"
        while True:
            try:
                with urllib.request.urlopen(url, timeout=10) as r:
                    buf = b""
                    while True:
                        buf += r.read1(65536)
                        while True:
                            i = buf.find(b"Content-Length: ")
                            j = buf.find(b"\r\n\r\n", i) if i >= 0 else -1
                            if i < 0 or j < 0:
                                break
                            n = int(buf[i + 16:buf.find(b"\r\n", i)])
                            if len(buf) < j + 4 + n:
                                break
                            jpg, buf = buf[j + 4:j + 4 + n], buf[j + 4 + n:]
                            self.live = time.time()
                            if self.viewers:
                                self.loop.call_soon_threadsafe(self.push, b"V" + jpg, True)
            except Exception:  # noqa: BLE001  (no run / bridge restarting)
                time.sleep(1)

    def audio_source(self):
        if self.a.audio_source != "auto":
            return self.a.audio_source
        sinks = subprocess.run(["pactl", "list", "short", "sinks"], capture_output=True, text=True).stdout
        return "phone_out.monitor" if "phone_out" in sinks else "@DEFAULT_MONITOR@"

    def audio_thread(self):
        while True:
            src = self.audio_source()
            p = subprocess.Popen(["parec", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                  "--latency-msec=20", "-d", src], stdout=subprocess.PIPE)
            t0 = time.time()
            while True:
                b = p.stdout.read(CHUNK * 2)
                if len(b) < CHUNK * 2:
                    break
                if self.viewers:
                    self.loop.call_soon_threadsafe(self.push, b"A" + b, False)
                if time.time() - t0 > 5 and src != self.audio_source():   # the run switched sinks: follow it
                    break
            p.kill()
            time.sleep(0.5)

    async def http(self, r, w):
        try:
            while True:
                head = await r.readuntil(b"\r\n\r\n")
                lines = head.decode(errors="replace").split("\r\n")
                method, path = (lines[0].split(" ") + ["", ""])[:2]
                hdrs = {ln.split(":", 1)[0].strip().lower(): ln.split(":", 1)[1].strip() for ln in lines[1:] if ":" in ln}
                await r.readexactly(int(hdrs.get("content-length", 0) or 0))
                ok = f"t={self.a.token}" in path or path.split("?")[0].rstrip("/") == f"/{self.a.token}"
                ctype, body, code = "text/plain", b"", "404 Not Found"
                if ok and path.startswith("/poll"):
                    vid = path.split("v=")[1].split("&")[0] if "v=" in path else "x"
                    v = self.viewers.setdefault(vid, {"q": collections.deque(), "t": time.time()})
                    v["t"] = time.time()
                    if "d=" in path and time.time() - v.get("logged", 0) > 5:
                        v["logged"] = time.time()
                        import urllib.parse
                        print(f"[monitor] viewer {vid}: {urllib.parse.unquote(path.split('d=')[1].split('&')[0])}", flush=True)
                    msgs = []
                    while v["q"]:
                        msgs.append(v["q"].popleft())
                    st = json.dumps({"live": time.time() - self.live < 2, "viewers": len(self.viewers)}).encode()
                    msgs.append(b"S" + st)
                    body = b"".join(struct.pack("<I", len(x)) + x for x in msgs)
                    ctype, code = "application/octet-stream", "200 OK"
                elif ok:
                    body, ctype, code = PAGE.replace("__TOKEN__", self.a.token).encode(), "text/html; charset=utf-8", "200 OK"
                w.write((f"HTTP/1.1 {code}\r\nContent-Type: {ctype}\r\nCache-Control: no-store\r\n"
                         f"Content-Length: {len(body)}\r\n\r\n").encode() + body)
                await w.drain()
        except (asyncio.IncompleteReadError, ConnectionError, OSError, asyncio.LimitOverrunError):
            pass
        finally:
            w.close()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="wlo1", help="listen only on this interface's address (the viewer's network)")
    ap.add_argument("--port", type=int, default=8450)
    ap.add_argument("--dashboard", type=int, default=8765)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--audio-source", default="auto", help="PulseAudio source (auto: phone_out.monitor if a phone "
                    "run is on, else the default output's monitor)")
    a = ap.parse_args()
    os.nice(10)                                          # never compete with the pipeline
    tf = REPO / ".tmp" / "monitor_token"
    if not tf.exists():
        tf.parent.mkdir(parents=True, exist_ok=True)
        tf.write_text(secrets.token_urlsafe(9))
    a.token = tf.read_text().strip()
    ip = iface_ip(a.iface)
    if not ip:
        sys.exit(f"[monitor] no IPv4 on {a.iface}")
    crt, key = server_cert([ip], REPO / ".tmp" / "monitor_cert")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(crt, key)
    m = Monitor(a)
    m.loop = asyncio.get_running_loop()
    threading.Thread(target=m.video_thread, daemon=True).start()
    threading.Thread(target=m.audio_thread, daemon=True).start()
    await asyncio.start_server(m.http, ip, a.port, ssl=ctx, limit=2**20)
    url = f"https://{ip}:{a.port}/{a.token}"
    (REPO / ".tmp" / "monitor_url").write_text(url)
    print(f"[monitor] watch on a phone on the same Wi-Fi ({a.iface}): {url}", flush=True)
    await asyncio.Future()


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><title>Monitor</title>
<style>[hidden]{display:none!important} html,body{margin:0;height:100%;background:#000;color:#ddd;font:15px -apple-system,system-ui}
#v{position:fixed;inset:0;width:100%;height:100%;object-fit:contain}
#bar{position:fixed;left:0;right:0;bottom:0;padding:10px max(12px,env(safe-area-inset-left)) calc(10px + env(safe-area-inset-bottom));
 display:flex;gap:10px;align-items:center;background:linear-gradient(transparent,#000c);font-size:13px}
#st{flex:1;text-align:right;opacity:.8} button{font:600 15px -apple-system;border:0;border-radius:20px;padding:9px 16px;background:#2e7d32;color:#fff}
#go{position:fixed;inset:0;display:flex;align-items:center;justify-content:center;background:#000}</style></head><body>
<img id="v" alt="" onload="D.img_ok++" onerror="D.img_err++"><div id="go"><button id="start">Watch (with sound)</button></div>
<button id="snd" hidden style="position:fixed;top:calc(12px + env(safe-area-inset-top));left:50%;transform:translateX(-50%);background:#c62828">Tap for sound</button>
<div id="bar" hidden><button id="mute">Sound on</button><span id="st"></span></div>
<script>
const T="__TOKEN__", VID=Math.random().toString(36).slice(2), $=id=>document.getElementById(id);
let ctx=null, playAt=0, frames=[], lastURL=null, muted=false, rx=0, live=false, okT=0, inflight=false;
const LAG=0.3; let D={img_ok:0,img_err:0,v:0,a:0,resets:0,ctx:""};
async function poll(){ if(inflight) return; inflight=true;
  try{ const r=await fetch(`/poll?t=${T}&v=${VID}&d=${encodeURIComponent(JSON.stringify({...D,ctx:ctx?ctx.state:"none",q:frames.length,buf:ctx?Math.round((playAt-ctx.currentTime)*1000):0}))}`,{method:"POST",cache:"no-store"}); const a=new Uint8Array(await r.arrayBuffer());
    okT=Date.now(); rx+=a.length; const dv=new DataView(a.buffer); let j=0;
    while(j+4<=a.length){ const n=dv.getUint32(j,true); onMsg(a.subarray(j+4,j+4+n)); j+=4+n; }
  }catch(e){ await new Promise(r=>setTimeout(r,500)); }
  inflight=false; }
setInterval(poll,30);
function onMsg(b){ const k=b[0];
  if(k===65&&ctx){ const s=new Int16Array(b.slice(1).buffer), buf=ctx.createBuffer(1,s.length,48000), d=buf.getChannelData(0);
    for(let i=0;i<s.length;i++)d[i]=muted?0:s[i]/32768;
    D.a++; const now=ctx.currentTime; if(playAt<now+0.04||playAt>now+0.6){ playAt=now+LAG; D.resets++; }
    const n=ctx.createBufferSource(); n.buffer=buf; n.connect(ctx.destination); n.start(playAt); playAt+=buf.duration; }
  else if(k===86){ D.v++; const wait=ctx&&ctx.state==="running"?Math.max(0,playAt-ctx.currentTime)+(ctx.outputLatency||0)+(ctx.baseLatency||0):0;
    frames.push([performance.now()+wait*1000, new Blob([b.subarray(1)],{type:"image/jpeg"})]); if(frames.length>40)frames.shift(); }
  else if(k===83){ try{ live=JSON.parse(new TextDecoder().decode(b.subarray(1))).live; }catch(_){} } }
function show(){ const now=performance.now(); let f=null; while(frames.length&&frames[0][0]<=now)f=frames.shift();
  if(f){ const u=URL.createObjectURL(f[1]); $("v").src=u; if(lastURL)URL.revokeObjectURL(lastURL); lastURL=u; }
  requestAnimationFrame(show); }
setInterval(()=>{ $("st").textContent=(Date.now()-okT>2000?"reconnecting…":(live?"live":"waiting for the stream"))+` · ${(rx*8/1e6/2).toFixed(1)} Mbit/s`; rx=0; },2000);
function unlock(){ if(!ctx) return; ctx.resume().catch(()=>{});
  // a silent buffer played inside the tap is what actually unlocks Web Audio on iOS
  try{ const b=ctx.createBuffer(1,1,48000), n=ctx.createBufferSource(); n.buffer=b; n.connect(ctx.destination); n.start(0); }catch(_){} }
document.addEventListener("touchend",unlock,{passive:true}); document.addEventListener("click",unlock);
setInterval(()=>{ if(ctx) $("snd").hidden=ctx.state==="running"; },500);
$("snd").onclick=unlock;
$("start").onclick=async()=>{ try{ if(navigator.audioSession) navigator.audioSession.type="playback"; }catch(_){}   // ignore the silent switch (iOS 17+)
  ctx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:48000}); unlock();
  $("go").hidden=true; $("bar").hidden=false; requestAnimationFrame(show); try{ await navigator.wakeLock.request("screen"); }catch(_){} };
$("mute").onclick=()=>{ muted=!muted; $("mute").textContent=muted?"Sound off":"Sound on"; if(ctx) ctx.resume().catch(()=>{}); };
document.addEventListener("visibilitychange",()=>{ if(document.visibilityState==="visible"&&ctx) ctx.resume().catch(()=>{}); });
</script></body></html>"""

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
