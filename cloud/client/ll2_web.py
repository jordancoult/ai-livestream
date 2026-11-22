"""Web UI client for the LongLive 2.0 live server (tools/longlive2/ll2_server.py, run on the box by ll2_live.sh).

  cloud/facefix/.venv/bin/python cloud/client/ll2_web.py --ssh "ssh -p PORT root@HOST"      then open http://127.0.0.1:8780

Everything is in the page: output (and webcam) views, prompt (smooth / hard cut), text or webcam mode, strength, person
keep, start image (assets or a pasted link: memory only, never saved), reset; server launch settings (resolution,
orientation, checkpoint, steps, attention, KV precision, clean pass, window, ahead, JPEG, re-cache) with "Restart server
with these settings" (the client reconnects by itself); local upscaling, sharpening, playout fps and buffer.
Optional extra outputs: --output /dev/video10 (virtual camera), ffplay, or an .mp4 path (comma-separated).
"""

import argparse
import asyncio
import base64
import io
import json
import os
import shlex
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--url", default="ws://127.0.0.1:8765")
p.add_argument("--ssh", default=None, help="ssh command to the GPU box; opens a tunnel to its port 8765")
p.add_argument("--ssh-key", default=os.path.expanduser("~/personal/ai-livestream/.secrets/runpod_ed25519"))
p.add_argument("--local-port", type=int, default=8765)
p.add_argument("--web-port", type=int, default=8780)
p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = also reachable from other devices on your network "
               "(phone: http://<laptop IP>:8780). No login: anyone on the network can control the stream")
p.add_argument("--input", default="/dev/video0", help="webcam (or a video file) used in v2v mode")
p.add_argument("--prompt", default="a cinematic shot of a cozy living room, warm light")
p.add_argument("--mode", default="t2v", choices=["t2v", "v2v"])
p.add_argument("--strength", type=float, default=0.9)
p.add_argument("--person", action="store_true")
p.add_argument("--output", default="", help="extra outputs: /dev/video10, ffplay, out.mp4 (comma-separated)")
a = p.parse_args()
if a.ssh and a.url == "ws://127.0.0.1:8765":  # connect through our own tunnel's local port
    a.url = f"ws://127.0.0.1:{a.local_port}"

ASSETS = Path(__file__).resolve().parents[2] / "assets"
LOCK = threading.Lock()
CFG = {"height": 704, "width": 1280}  # server config (from its "config" message)
ST = {"prompt": a.prompt, "mode": a.mode, "strength": a.strength, "person": a.person, "spread": False, "status": "starting",
      "gen_fps": None, "gen_ms": None, "ahead": None, "change_latency": None, "message": "", "version": 0,
      "played": 0, "repeated": 0, "snapped": 0}
LOCAL = {"upscale": 0, "sharp": 0.0, "fps": "auto", "buffer": 4}
Q = deque()  # (frame bgr, version)
VIEWS = {"out": None, "in": None}  # latest frames for the page
VIEW_COND = threading.Condition()
VIEW_JPG = {"out": (0, None), "in": (0, None)}  # (serial, jpeg bytes)


def publish(key, fr, q=88):
    ok, jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, q])
    with VIEW_COND:
        VIEW_JPG[key] = (VIEW_JPG[key][0] + 1, jpg.tobytes())
        VIEW_COND.notify_all()
SENT_T = {}
CMDS: asyncio.Queue = None
LOOP = None
STOP = threading.Event()


def size():
    return CFG["width"], CFG["height"]


def log(msg):
    print(f"[ll2_web] {msg}", flush=True)
    ST["message"] = msg


# ---------------------------------------------------------------- tunnel + input
def open_tunnel():
    if not a.ssh:
        return None
    parts = shlex.split(a.ssh)
    cmd = parts[:1] + ["-i", a.ssh_key, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                       "-o", "LogLevel=ERROR", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15",
                       "-N", "-L", f"{a.local_port}:127.0.0.1:8765"] + parts[1:]
    t = subprocess.Popen(cmd)
    time.sleep(3)
    if t.poll() is not None:
        raise SystemExit("[ll2_web] ssh tunnel failed")
    return t


class Input:
    """Webcam (opened only while in v2v mode) or a looped file, center-cropped to the server size."""
    def __init__(self, src):
        self.src, self.cap, self.frame, self.on = src, None, None, False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not STOP.is_set():
            if not self.on:
                if self.cap is not None:
                    self.cap.release()
                    self.cap, self.frame = None, None
                time.sleep(0.1)
                continue
            if self.cap is None:
                s = self.src
                self.cap = cv2.VideoCapture(int(s[10:]) if s.startswith("/dev/video") and s[10:].isdigit() else s)
                self.is_file = not s.startswith("/dev/video")
                self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            ok, f = self.cap.read()
            if not ok:
                if self.is_file:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                else:
                    time.sleep(0.01)
                continue
            W, H = size()
            h, w = f.shape[:2]
            s = max(W / w, H / h)
            f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            y0, x0 = (f.shape[0] - H) // 2, (f.shape[1] - W) // 2
            self.frame = f[y0:y0 + H, x0:x0 + W]
            VIEWS["in"] = self.frame
            if self.is_file:
                time.sleep(1.0 / self.fps)


INP = Input(a.input)


class Segmenter:
    def __init__(self):
        import mediapipe as mp
        self.seg = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def __call__(self, bgr):
        m = self.seg.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).segmentation_mask
        return (np.clip(m, 0, 1) * 255).astype(np.uint8)


SEG = {"seg": None}


# ---------------------------------------------------------------- start images (memory only)
START = {"raw": None, "label": ""}  # current start image: raw bytes in RAM only (never written to disk, here or on the box)


def fetch_image(src: str):
    """An assets/ file name, an http(s) link (Google Images result links unwrapped) or a data: URL -> (raw bytes,
    label). Memory only."""
    src = src.strip()
    if src.startswith("data:"):
        raw = base64.b64decode(src.split(",", 1)[1])
    elif src.startswith(("http://", "https://")):
        q = urllib.parse.urlparse(src)
        if "google." in q.netloc:
            qs = urllib.parse.parse_qs(q.query)
            for k in ("imgurl", "url", "q"):
                if qs.get(k, [""])[0].startswith("http"):
                    src = qs[k][0]
                    break
        req = urllib.request.Request(src, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
                                                   "Accept": "image/*,text/html;q=0.9,*/*;q=0.8"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read(25 * 2**20 + 1)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"download failed ({getattr(exc, 'code', '') or type(exc).__name__}): "
                             f"{getattr(exc, 'reason', exc)}")
        if len(raw) > 25 * 2**20:
            raise ValueError("image larger than 25 MB")
        if raw.lstrip()[:1] == b"<":  # a web page (e.g. a phone's Share link): use its preview image (og:image)
            import html as _html
            import re
            page = raw[:2_000_000].decode("utf-8", "ignore")
            m = None
            for pat in (r'<meta[^>]+(?:property|name)=["\'](?:og:image(?::secure_url|:url)?|twitter:image(?::src)?)["\'][^>]*>',
                        r'<link[^>]+rel=["\']image_src["\'][^>]*>'):
                tag = re.search(pat, page, re.I)
                if tag:
                    m = re.search(r'(?:content|href)=["\']([^"\']+)["\']', tag.group(0), re.I)
                    if m:
                        break
            if not m:
                raise ValueError("that link is a web page without a preview image: open the image itself and copy its "
                                 "address, or use Upload / choose photo")
            img_url = urllib.parse.urljoin(src, _html.unescape(m.group(1)))
            req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)",
                                                           "Accept": "image/*,*/*", "Referer": src})
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    raw = r.read(25 * 2**20 + 1)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"preview image download failed ({getattr(exc, 'code', '') or type(exc).__name__})")
            if len(raw) > 25 * 2**20:
                raise ValueError("image larger than 25 MB")
        label = "link: " + urllib.parse.urlparse(src).netloc
    else:
        raw = (ASSETS / Path(src).name).read_bytes()
        label = Path(src).name
    if src.startswith("data:"):
        label = "uploaded image"
    crop_image(raw)  # validate now (readable error)
    return raw, label


def crop_image(raw: bytes) -> bytes:
    """Raw image bytes -> PNG centre-cropped to the current server aspect and resized to its size (in memory)."""
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise ValueError("not an image (for Google Images, open the image and copy its address)")
    W, H = size()
    iw, ih = img.size
    if iw / ih > W / H:
        cw = round(ih * W / H)
        img = img.crop(((iw - cw) // 2, 0, (iw - cw) // 2 + cw, ih))
    else:
        ch = round(iw * H / W)
        img = img.crop((0, (ih - ch) // 2, iw, (ih - ch) // 2 + ch))
    buf = io.BytesIO()
    img.resize((W, H), Image.LANCZOS).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------- outputs
class Sinks:
    def __init__(self):
        self.procs, self.wh = [], None

    def write(self, fr):
        wh = (fr.shape[1], fr.shape[0])
        if wh != self.wh:
            self.close()
            self.wh = wh
            for o in [o for o in a.output.split(",") if o]:
                base = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{wh[0]}x{wh[1]}",
                        "-r", "25", "-i", "-"]
                if o == "ffplay":
                    cmd = ["ffplay", "-v", "error", "-window_title", "LongLive 2.0", "-f", "rawvideo", "-pixel_format",
                           "bgr24", "-video_size", f"{wh[0]}x{wh[1]}", "-fflags", "nobuffer", "-"]
                elif o.startswith("/dev/video"):
                    cmd = base + ["-pix_fmt", "yuv420p", "-f", "v4l2", o]
                else:
                    cmd = base + ["-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", o]
                self.procs.append(subprocess.Popen(cmd, stdin=subprocess.PIPE))
        for pr in self.procs:
            try:
                pr.stdin.write(fr.tobytes())
            except (BrokenPipeError, ValueError):
                pass

    def close(self):
        for pr in self.procs:
            try:
                pr.stdin.close()
                pr.wait(timeout=5)
            except Exception:
                pr.kill()
        self.procs = []


SINKS = Sinks()


def post_process(fr):
    u, sh = int(LOCAL["upscale"] or 0), float(LOCAL["sharp"] or 0)
    if u and u > fr.shape[0]:
        fr = cv2.resize(fr, (round(fr.shape[1] * u / fr.shape[0]), u), interpolation=cv2.INTER_LANCZOS4)
    if sh > 0:
        blur = cv2.GaussianBlur(fr, (0, 0), 1.2)
        fr = cv2.addWeighted(fr, 1 + sh, blur, -sh, 0)
    return fr


async def player():
    last = None
    shown = 0
    nxt = time.time()
    started = False
    while not STOP.is_set():
        if LOCAL["fps"] == "auto":
            fps = CFG.get("fps", 24)
        elif LOCAL["fps"] == "match":  # play at the generation rate: smooth, no once-per-chunk stall
            fps = max(8.0, min(float(ST["gen_fps"] or CFG.get("fps", 24)) * 0.97, CFG.get("fps", 24)))
            if len(Q) > 48:  # a chunk and a half queued: catch up a little (never above the server rate)
                fps = min(fps * 1.05, CFG.get("fps", 24))
        else:
            fps = float(LOCAL["fps"])
        if not started:
            if len(Q) >= int(LOCAL["buffer"]):
                started, nxt = True, time.time()
            else:
                await asyncio.sleep(0.005)
                continue
        if Q:
            last, ver = Q.popleft()
            ST["played"] += 1
            if ver > shown:
                if ver in SENT_T and ver > 1:
                    ST["change_latency"] = round(time.time() - SENT_T[ver], 2)
                shown = ver
        else:
            ST["repeated"] += 1
        if last is not None:
            fr = await asyncio.to_thread(post_process, last) if (LOCAL["upscale"] or LOCAL["sharp"]) else last
            VIEWS["out"] = fr
            await asyncio.to_thread(publish, "out", fr)
            SINKS.write(fr)
        nxt += 1.0 / fps
        await asyncio.sleep(max(0, nxt - time.time()))


# ---------------------------------------------------------------- server session
def server_args(s: dict) -> str:
    """Page form -> ll2_server.py launch args."""
    args = [f"--ckpt {s.get('ckpt', 'S2')}", f"--height {int(s['height'])}", f"--width {int(s['width'])}"]
    if s.get("steps"):
        args.append(f"--steps {int(s['steps'])}")
    if s.get("sage"):
        args.append("--sage")
    if s.get("kvq", True):
        args.append("--kvq")
    for k, flag in (("clean_every", "--clean-every"), ("window", "--window"), ("ahead", "--ahead"), ("overlap", "--overlap"),
                    ("jpeg", "--jpeg"), ("recache_frames", "--recache-frames")):
        if s.get(k) not in (None, ""):
            args.append(f"{flag} {int(s[k])}")
    if s.get("rope") == "relative":
        args.append("--rope relative")
    return " ".join(args)


async def session(ws):
    try:  # the server sends its config first: crop the start image to its (possibly new) size
        j = json.loads(await asyncio.wait_for(ws.recv(), 10))
        if j.get("type") == "config":
            if (j["width"], j["height"]) != size():
                Q.clear()
            CFG.update(j)
            ST["status"] = "streaming"
            log(f"server: {j['args']}")
    except (asyncio.TimeoutError, ValueError, TypeError):
        pass
    start = {"type": "start", "prompt": ST["prompt"], "mode": ST["mode"], "strength": ST["strength"], "person": ST["person"]}
    await ws.send(json.dumps({"type": "set", "spread": ST["spread"]}))
    if START["raw"] is not None:  # re-apply the start image after a server restart (from RAM)
        start["image"] = base64.b64encode(crop_image(START["raw"])).decode()
    await ws.send(json.dumps(start))
    ST["version"] += 1
    SENT_T[ST["version"]] = time.time()
    done = asyncio.Event()

    async def receiver():
        async for m in ws:
            if isinstance(m, bytes):
                seq, ver, chunk, src = struct.unpack("<IIII", m[:16])
                img = cv2.imdecode(np.frombuffer(m[16:], np.uint8), cv2.IMREAD_COLOR)
                if Q and ver > Q[-1][1]:  # snap: drop frames of the old prompt as the new one arrives
                    ST["snapped"] += len(Q)
                    Q.clear()
                Q.append((img, ver))
            else:
                j = json.loads(m)
                t = j.get("type")
                if t == "config":
                    if (j["width"], j["height"]) != size():
                        Q.clear()
                    CFG.update(j)
                    ST["status"] = "streaming"
                    log(f"server: {j['args']}")
                elif t == "stats":
                    ST["gen_fps"], ST["gen_ms"], ST["ahead"] = j.get("gen_fps"), j.get("gen_ms"), j.get("ahead_frames")
                elif t == "restarting":
                    ST["status"] = "server restarting (warm-up ~1 min)"
        done.set()

    async def sender():  # webcam frames in v2v
        period = 1 / 24
        nxt = time.time()
        while not done.is_set():
            INP.on = ST["mode"] == "v2v"
            f = INP.frame
            if INP.on and f is not None:
                ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 85])
                await ws.send(b"F" + struct.pack("<I", 0) + jpg.tobytes())
                if ST["person"]:
                    if SEG["seg"] is None:
                        SEG["seg"] = Segmenter()
                    ok, mj = cv2.imencode(".jpg", SEG["seg"](f), [cv2.IMWRITE_JPEG_QUALITY, 80])
                    await ws.send(b"M" + mj.tobytes())
            nxt += period
            await asyncio.sleep(max(0, nxt - time.time()))

    async def reporter():  # the server paces generation on frames consumed
        while not done.is_set():
            await ws.send(json.dumps({"type": "played", "n": ST["played"] + ST["snapped"]}))
            await asyncio.sleep(0.1)

    async def commands():
        while not done.is_set():
            try:
                c = await asyncio.wait_for(CMDS.get(), 0.3)
            except asyncio.TimeoutError:
                continue
            msg = None
            if "prompt" in c:
                ST["prompt"] = c["prompt"]
                msg = {"type": "prompt", "text": c["prompt"], "cut": bool(c.get("cut"))}
            elif "set" in c:
                for k in ("mode", "strength", "person", "spread"):
                    if k in c["set"]:
                        ST[k] = c["set"][k]
                msg = {"type": "set", **{k: ST[k] for k in ("mode", "strength", "person", "spread")}}
            elif "reset" in c:
                msg = {"type": "reset", "prompt": ST["prompt"]}
            elif "image" in c:
                msg = {"type": "image", "data": base64.b64encode(c["image"]).decode(), "prompt": ST["prompt"]}
            elif "server" in c:
                msg = {"type": "server", "args": c["server"]}
                ST["status"] = "server restarting (warm-up ~1 min)"
            if msg is None:
                continue
            if msg["type"] in ("prompt", "image", "reset"):
                ST["version"] += 1
                SENT_T[ST["version"]] = time.time()
            await ws.send(json.dumps(msg))
            log(f"-> {msg['type']}: {str(c.get('prompt') or c.get('server') or c.get('set') or '')[:90]}")

    tasks = [asyncio.ensure_future(x()) for x in (receiver, sender, reporter, commands)]
    await done.wait()
    for t in tasks:
        t.cancel()


async def session_loop():
    import websockets

    while not STOP.is_set():
        try:
            ws = await websockets.connect(a.url, max_size=2 ** 25, ping_interval=20, open_timeout=5)
        except Exception:
            if not ST["status"].startswith("server restarting"):
                ST["status"] = "connecting to the server..."
            await asyncio.sleep(2)
            continue
        try:
            await session(ws)
        except Exception as exc:  # noqa: BLE001
            log(f"session ended: {exc!r}")
        finally:
            try:
                await ws.close()
            except Exception:
                pass
        if not ST["status"].startswith("server restarting"):
            ST["status"] = "reconnecting..."
        await asyncio.sleep(2)


# ---------------------------------------------------------------- web
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LongLive 2.0 live</title><style>
:root{--bg:#0f1115;--fg:#e7e9ee;--mut:#9aa3b2;--card:#181b22;--line:#2a2f3a;--acc:#5aa9ff;--warn:#f0b357;--bad:#ff6b6b;--ok:#5fd38d}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(320px,1fr);gap:14px;padding:14px;max-width:1700px;margin:auto}
@media(max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:12px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:0 0 8px}
img.view{width:100%;border-radius:8px;background:#000;display:block}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}
label{color:var(--mut)} select,input,textarea,button{font:inherit;color:var(--fg);background:#10131a;border:1px solid var(--line);border-radius:6px;padding:5px 8px}
textarea{width:100%;box-sizing:border-box} input[type=range]{flex:1;min-width:120px}
button{cursor:pointer;background:#1f2633} button.pri{background:var(--acc);color:#071222;border-color:var(--acc);font-weight:600}
button.warn{background:#3a2c14;border-color:var(--warn);color:var(--warn)}
.stat{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;font-variant-numeric:tabular-nums}
.stat b{font-weight:500;color:var(--mut)} .ok{color:var(--ok)} .bad{color:var(--bad)} .warn{color:var(--warn)}
.tip{display:inline-block;margin-left:5px;width:15px;height:15px;line-height:15px;text-align:center;border-radius:50%;
 background:#2a3242;color:var(--fg);font-size:11px;cursor:help;position:relative}
.tip:hover::after,.tip:focus::after{content:attr(data-tip);position:absolute;left:20px;top:-6px;z-index:10;width:340px;white-space:pre-line;
 background:#0b0d12;border:1px solid var(--line);border-radius:8px;padding:9px 11px;font-size:12.5px;line-height:1.45;color:var(--fg);
 box-shadow:0 6px 24px #0008;text-align:left}
@media(max-width:900px){.tip:hover::after,.tip:focus::after{left:auto;right:0;top:20px;width:min(340px,80vw)}}
.grid2{display:grid;grid-template-columns:auto 1fr;gap:6px 10px;align-items:center} .muted{color:var(--mut);font-size:12px}
</style></head><body><main>
<section>
 <div class="card"><h2>Output</h2><img class="view" id="out" src="/stream/out"></div>
 <div class="card" id="incard"><h2>Input (webcam mode)</h2><img class="view" id="in" style="max-width:50%"></div>
</section>
<section>
 <div class="card"><h2>Status</h2><div class="stat" id="stat"></div></div>
 <div class="card"><h2>Prompt</h2>
  <textarea id="prompt" rows="3"></textarea>
  <div class="row"><button class="pri" id="smooth">Change (smooth, ~2 s)</button><button id="cut">Hard cut</button><button id="reset">Restart stream</button></div>
  <div class="muted">Enter = smooth change. A hard cut clears the scene memory.</div></div>
 <div class="card"><h2>Input</h2>
  <div class="row"><label>Mode</label><select id="mode"><option value="t2v">Text only</option><option value="v2v">Webcam (video to video)</option></select>
   <label><input type="checkbox" id="person"> Keep me (person)</label>
   <label title="When generation is slower than real time, each chunk normally takes your latest frames, so the output jumps forward to catch up. Smooth spreads the chunk's frames over everything since the last chunk: continuous but slightly fast-forwarded motion."><input type="checkbox" id="spread"> Smooth motion when slow</label></div>
  <div class="row"><label>Strength</label><input type="range" id="strength" min="0.5" max="1" step="0.01"><span id="strv"></span></div>
  <div class="muted">Webcam strength: 0.85 keeps your room (faster), 0.95+ new scene (slower). Person keep: try 0.98.</div>
  <div class="row"><label>Start image</label><select id="img"></select><button id="imggo">Start from it</button></div>
  <div class="row"><input id="imgurl" placeholder="or paste an image link (Google Images ok) + Enter: memory only, never saved" style="flex:1"></div>
  <div class="row"><label class="btn" for="imgfile" style="border:1px solid var(--line);border-radius:6px;padding:5px 8px;background:#1f2633;color:var(--fg);cursor:pointer">Upload / choose photo</label>
   <input type="file" id="imgfile" accept="image/*" style="display:none"><span class="muted">or paste an image anywhere on the page</span></div>
  <div class="row"><span class="muted" id="startimg"></span><button id="imgclear">Clear</button></div></div>
 <div class="card"><h2>Server settings</h2><div class="grid2">
  <label>Resolution</label><select id="res"><option value="1280x704">1280×704 (native, 720p)</option><option value="1024x576">1024×576 (fast)</option><option value="960x544">960×544</option><option value="832x480">832×480</option></select>
  <label>Orientation</label><select id="ori"><option value="land">Landscape</option><option value="port">Portrait</option></select>
  <label>Checkpoint</label><select id="ckpt"><option value="S2">S2 (2-step)</option><option value="S4">S4 (4-step, richer, slower)</option></select>
  <label>Steps</label><select id="steps"><option value="">checkpoint default</option><option>1</option><option>2</option><option>3</option><option>4</option><option>6</option><option>8</option></select>
  <label>Attention</label><select id="sage"><option value="0">Exact (best quality)</option><option value="1">SageAttention (fast, may smear)</option></select>
  <label>KV cache</label><select id="kvq"><option value="1">4-bit NVFP4 (fits 32 GB)</option><option value="0">bf16 (faster, may run out of memory)</option></select>
  <label>Clean pass</label><select id="clean_every"><option value="1">every chunk (best)</option><option value="2">every 2nd (faster)</option><option value="3">every 3rd</option></select>
  <label>Memory window</label><select id="window"><option>32</option><option>28</option><option>24</option><option>20</option><option>16</option></select>
  <label>Chunk</label><select id="overlap"><option value="0">8 latents / 32 frames (stock)</option><option value="4">overlap 4: 16 frames (half latency, 2x compute)</option><option value="2">overlap 2: 24 frames</option></select>
  <label>Work ahead</label><select id="ahead"><option value="-1">auto (just in time, lowest latency)</option><option value="34">34 frames</option><option value="48">48 frames</option><option value="64">64 frames (smooth, +1.3 s)</option></select>
  <label>JPEG quality</label><select id="jpeg"><option>95</option><option>90</option><option>80</option><option>70</option></select>
  <label>Prompt re-cache</label><select id="recache_frames"><option value="16">16 frames (default)</option><option value="32">32 (stronger, slower)</option><option value="8">8</option><option value="0">off (fast, weak change)</option></select>
 <label>Positions</label><select id="rope"><option value="absolute">absolute (upstream)</option><option value="relative">window-relative (experimental)</option></select>
 </div><div class="row"><button class="warn" id="restart">Restart server with these settings</button></div>
 <div class="muted">Restart takes ~1 min (warm-up). If a setting crashes the server (e.g. out of memory) it falls back to the last working settings.</div></div>
 <div class="card"><h2>Local (instant)</h2><div class="grid2">
  <label>Upscale</label><select id="upscale"><option value="0">off (native)</option><option value="1080">1080p (Lanczos)</option><option value="1440">1440p (Lanczos)</option></select>
  <label>Sharpen</label><div class="row" style="margin:0"><input type="range" id="sharp" min="0" max="1.5" step="0.05"><span id="shv"></span></div>
  <label>Playout fps</label><select id="fps"><option value="auto">server (24)</option><option value="match">match generation (smooth, no stalls)</option><option>25</option><option>24</option><option>20</option><option>18</option><option>15</option></select>
  <label>Buffer</label><select id="buffer"><option>4</option><option>8</option><option>16</option><option>32</option></select>
 </div><div class="muted">If generation fps is below playout fps, playback stutters once per chunk: lower the playout fps or speed up the server.</div></div>
</section></main>
<script>
const $=id=>document.getElementById(id);
const TIPS={"res": "What: the size the model generates at (latent grid = width/16 \u00d7 height/16; 1280\u00d7704 = 80\u00d744 = 3,520 tokens per latent frame).\nArchitecture: Wan2.2-5B was trained at 1280\u00d7704, so native is in-distribution; smaller sizes are slightly out of distribution but still work.\nTradeoff: compute grows with tokens (attention a bit faster than linearly). 1280\u00d7704 \u2248 18 fps on a 5090, 1024\u00d7576 \u2248 33 fps. Native = sharpest detail; lower = speed headroom.", "ori": "What: swaps width and height (e.g. 704\u00d71280 for phone-style vertical video).\nArchitecture: the same token count, so the same speed; the model saw far fewer portrait clips in training.\nTradeoff: expect composition/anatomy to be a bit weaker in portrait (untested here). Webcam input is centre-cropped to match.", "ckpt": "What: which distilled checkpoint is loaded. S2 = distilled for 2 denoising steps per chunk, S4 = for 4.\nArchitecture: both are causal (autoregressive) students distilled from Wan2.2-TI2V-5B, NVFP4-quantized; each step is one full transformer pass over the 8-latent block.\nTradeoff: S4 = richer texture/detail and steadier faces, but ~1.7\u00d7 the DiT time per chunk (needs Sage or a faster GPU to stay above 24 fps). S2 = the speed default.", "steps": "What: overrides the checkpoint's number of denoising steps per chunk.\nArchitecture: each step = one transformer pass over the block (plus the clean pass at the end). The distilled timestep schedule is matched to the checkpoint's default.\nTradeoff: more steps than the checkpoint was distilled for gives little and costs linearly; fewer steps = faster but blurrier/less coherent. Leave on default unless experimenting.", "sage": "What: the attention kernel. Exact = FlashAttention in bf16. Sage = SageAttention 2 (INT8 Q\u00b7K, FP8 P\u00b7V).\nArchitecture: attention runs over the current block + the KV window (up to 32 latent frames \u00d7 3,520 tokens) in every one of the 30 layers: the biggest cost at native.\nTradeoff: Sage \u2248 +40-50% fps (native \u2248 27-28 fps) but the quantization error shows as smeared faces in motion and artifacts at 1280\u00d7704. Exact = clean.", "kvq": "What: precision of the KV cache (the stored keys/values of past frames the model attends to).\nArchitecture: 30 layers \u00d7 window \u00d7 3,520 tokens/frame of K and V. NVFP4 (4-bit with per-block scales) is how upstream ships it.\nTradeoff: bf16 KV is slightly faster (no quantize/dequantize) and marginally more exact, but uses ~4\u00d7 the cache memory: at native with the T5 on the GPU it may not fit in 32 GB (the server falls back if it crashes).", "clean_every": "What: how often the finished chunk is re-run through the model at noise level 0 to write its keys/values into the cache.\nArchitecture: LongLive/Self-Forcing style: the denoising passes see noisy versions of the chunk; the clean pass stores what the chunk 'really' is, so later chunks condition on clean context.\nTradeoff: every 2nd chunk saves one full pass on half the chunks (\u2248 +15-20% fps), but the skipped chunks leave noisier context in memory: less consistency, more drift. Every chunk = best quality.", "window": "What: how many latent frames of past context the model attends to (each latent frame = 4 video frames; 32 = ~5.3 s at 24 fps), including the 8-frame attention sink.\nArchitecture: sliding-window causal attention: the oldest frames roll out, the first chunk stays pinned as the 'sink' (an anchor that stabilizes attention).\nTradeoff: smaller = faster attention but the scene forgets sooner and drifts; larger than trained (32) doesn't help (tested). Note: identity/scene memory beyond the window is lost.", "overlap": "What: how many video frames each step adds. Stock: an 8-latent block = 32 frames (1.33 s). Overlap 4 (lookahead): still an 8-latent block, but only the first 4 latents (16 frames) are shown; the other 4 are regenerated next step.\nArchitecture: the model only works on its trained 8-latent block, so the cost per step stays the same.\nTradeoff: halves the 'wait for a chunk' latency (-0.67 s) but halves the fps (\u224810 fps at native on a 5090) and text mode gets artifacts. Only useful on a much faster GPU.", "ahead": "What: how many unplayed frames the server keeps queued at your client before it starts the next chunk.\nArchitecture: generation is paced on frames you've actually played (reported back), not a server clock.\nTradeoff: auto = just-in-time (measured generation time + 6 frames): lowest latency, but any hiccup shows as a short stall. Larger = smoother through jitter, +1 s of latency per 24 frames. Only matters when generation is faster than playout.", "jpeg": "What: JPEG quality of the frames sent from the box to you.\nArchitecture: frames are encoded on the box's CPU (in a separate thread, streamed frame by frame) and decoded on the laptop.\nTradeoff: at 1280\u00d7704: q90 \u2248 120 KB/frame (\u224824 Mbit/s at 25 fps), q80 \u2248 80 KB (\u224817 Mbit/s), q70 \u2248 65 KB. Higher = fewer block artifacts but longer bursts on the network; q80 is visually near-lossless at 720p.", "recache_frames": "What: on a smooth prompt change, how many recent latent frames get their keys/values rebuilt under the new prompt.\nArchitecture: LongLive's KV-recache: the cached context was computed with the old prompt's cross-attention; without a rebuild the old scene dominates and the new prompt barely takes hold.\nTradeoff: more frames = the new prompt takes hold faster and stronger, but a one-off extra cost (a hitch) at the change; 0 = no hitch, the change is weak and slow. The sink is never rebuilt (use a hard cut for that).", "rope": "What: how frame positions are encoded (rotary position embeddings in time).\nArchitecture: absolute = upstream: every frame gets its true position (table extended so streams run for hours). Window-relative = the sink + window are laid out contiguously every chunk, so positions stay bounded.\nTradeoff: relative sounds ideal for infinite streams, but the model wasn't trained that way: in tests it drifted sooner. Keep absolute unless experimenting."};
for(const [id,t] of Object.entries(TIPS)){ const el=document.getElementById(id); if(!el) continue; const lab=el.previousElementSibling;
 if(lab&&lab.tagName==='LABEL'){ const q=document.createElement('span'); q.className='tip'; q.tabIndex=0; q.textContent='?'; q.dataset.tip=t; lab.appendChild(q); } } let S=null, formInit=false;
const post=(u,b)=>fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)}).then(r=>r.json());
function msg(t){S&&(S.message=t)}
function stat(){ if(!S) return; const c=S.cfg, pf=S.local.fps==='auto'?(c.fps||24):S.local.fps==='match'?Math.min(c.fps||24,+((S.gen_fps||24)*0.97).toFixed(1)):+S.local.fps, g=S.gen_fps;
 const gcls=g==null?'':(g>=pf*1.03?'ok':g>=pf*0.97?'warn':'bad');
 $('stat').innerHTML=`<b>Status</b><span>${S.status}</span><b>Server</b><span>${c.width}×${c.height}, ${c.ckpt||''}${c.steps?' '+c.steps+' steps':''}, ${c.sage?'Sage':'exact'} attention, KV ${c.kvq?'4-bit':'bf16'}, clean every ${c.clean_every}, window ${c.window}</span>
 <b>Generation</b><span class="${gcls}">${g??'–'} fps</span><b>Playout</b><span>${pf} fps · buffer ${S.buffer} · repeated ${S.repeated}</span>
 <b>Last change</b><span>${S.change_latency??'–'} s</span><b>Last</b><span class="muted">${S.message||''}</span>`;
 $('incard').style.display=S.mode==='v2v'?'':'none'; if(S.mode==='v2v' && !$('in').src.includes('/stream/in')) $('in').src='/stream/in?'+Date.now();
 if(!formInit){ formInit=true; const c2=S.cfg; $('prompt').value=S.prompt; $('mode').value=S.mode; $('person').checked=S.person; $('spread').checked=!!S.spread; $('strength').value=S.strength;
  const port=c2.height>c2.width, w=port?c2.height:c2.width, h=port?c2.width:c2.height; $('res').value=`${w}x${h}`; $('ori').value=port?'port':'land';
  $('ckpt').value=c2.ckpt||'S2'; $('steps').value=c2.steps||''; $('sage').value=c2.sage?'1':'0'; $('kvq').value=c2.kvq===false?'0':'1';
  $('clean_every').value=c2.clean_every||1; $('window').value=c2.window||32;
  const ar=(c2.args||'').match(/--ahead (\d+)/), jq=(c2.args||'').match(/--jpeg (\d+)/), rc=(c2.args||'').match(/--recache-frames (\d+)/);
  $('ahead').value=ar?ar[1]:'-1'; $('overlap').value=c2.overlap||0; $('jpeg').value=jq?jq[1]:'90'; $('recache_frames').value=rc?rc[1]:'16'; $('rope').value=c2.rope||'absolute';
  $('upscale').value=S.local.upscale; $('sharp').value=S.local.sharp; $('fps').value=S.local.fps; $('buffer').value=S.local.buffer;
  $('img').innerHTML=S.assets.map(f=>`<option>${f}</option>`).join(''); }
 $('startimg').textContent=S.start_image?('Start image: '+S.start_image+' (kept in memory, re-applied after restarts)'):'No start image (text start)'; $('imgclear').style.display=S.start_image?'':'none';
 $('strv').textContent=(+$('strength').value).toFixed(2); $('shv').textContent=(+$('sharp').value).toFixed(2); }
let down=false;
function reattach(){ const t=Date.now(); $('out').src='/stream/out?'+t; if(S&&S.mode==='v2v') $('in').src='/stream/in?'+t; }
async function poll(){ try{ S=await (await fetch('/state')).json(); if(down){ down=false; formInit=false; reattach(); } stat(); }
 catch(e){ if(!down){ down=true; if(S) S.status='local script not running (waiting for it...)'; stat(); } } }
$('out').onerror=()=>setTimeout(()=>{ if(!down) $('out').src='/stream/out?'+Date.now(); },1000);
$('in').onerror=()=>setTimeout(()=>{ if(!down&&S&&S.mode==='v2v') $('in').src='/stream/in?'+Date.now(); },1000);
setInterval(poll,700); poll();
$('smooth').onclick=()=>post('/cmd',{prompt:$('prompt').value});
$('cut').onclick=()=>post('/cmd',{prompt:$('prompt').value,cut:true});
$('reset').onclick=()=>post('/cmd',{reset:true});
$('prompt').onkeydown=e=>{ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); $('smooth').click(); } };
$('mode').onchange=()=>post('/cmd',{set:{mode:$('mode').value}});
$('spread').onchange=()=>post('/cmd',{set:{spread:$('spread').checked}});
$('person').onchange=()=>post('/cmd',{set:{person:$('person').checked}});
$('strength').onchange=()=>post('/cmd',{set:{strength:+$('strength').value}}); $('strength').oninput=stat;
function sendImageFile(f){ if(!f||!f.type.startsWith('image/')) return; if(f.size>25e6){ alert('Image larger than 25 MB'); return; }
 const r=new FileReader(); r.onload=()=>post('/cmd',{image:r.result}).then(x=>{ if(!x.ok) alert('Image not used: '+x.error); }); r.readAsDataURL(f); }
$('imgfile').onchange=e=>{ sendImageFile(e.target.files[0]); e.target.value=''; };
document.addEventListener('paste',e=>{ if(e.target&&e.target.id==='prompt') return; const it=[...(e.clipboardData||{}).items||[]].find(i=>i.type.startsWith('image/'));
 if(it){ e.preventDefault(); sendImageFile(it.getAsFile()); } });
$('imgclear').onclick=()=>post('/cmd',{clear_image:true});
$('imggo').onclick=()=>post('/cmd',{image:$('img').value}).then(r=>!r.ok&&alert(r.error));
$('imgurl').onkeydown=e=>{ if(e.key==='Enter'&&$('imgurl').value.trim()) post('/cmd',{image:$('imgurl').value.trim()}).then(r=>{ if(r.ok) $('imgurl').value=''; else alert('Image not used: '+r.error); }); };
$('restart').onclick=()=>{ const [w,h]=$('res').value.split('x').map(Number), port=$('ori').value==='port';
 const s={height:port?w:h,width:port?h:w,ckpt:$('ckpt').value,steps:$('steps').value,sage:$('sage').value==='1',kvq:$('kvq').value==='1',
  clean_every:$('clean_every').value,window:$('window').value,ahead:$('ahead').value,overlap:$('overlap').value,jpeg:$('jpeg').value,recache_frames:$('recache_frames').value,rope:$('rope').value};
 if(confirm('Restart the server with these settings? The stream stops for ~1 min.')) post('/server',s).then(()=>{formInit=false;}); };
for(const id of ['upscale','sharp','fps','buffer']) $(id).onchange=()=>post('/local',{[id]:$(id).value}); $('sharp').oninput=stat;
</script></body></html>"""


class Web(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            b = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif self.path == "/state":
            assets = sorted(f.name for f in ASSETS.glob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))
            self._json({**ST, "cfg": CFG, "local": LOCAL, "buffer": len(Q), "assets": assets, "start_image": START["label"]})
        elif self.path.startswith("/stream/"):
            key = self.path.split("/")[2].split("?")[0]
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            seen = 0
            try:
                while not STOP.is_set():
                    if key == "in":  # webcam preview: ~15 fps is plenty
                        fr = VIEWS.get("in")
                        if fr is not None:
                            publish("in", fr, 80)
                        time.sleep(1 / 15)
                    with VIEW_COND:
                        VIEW_COND.wait_for(lambda: VIEW_JPG[key][0] != seen or STOP.is_set(), timeout=1.0)
                        seen, jpg = VIEW_JPG[key]
                    if jpg is None:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                                     str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/cmd":
            if "image" in body:
                try:
                    raw, label = fetch_image(str(body["image"]))
                except Exception as exc:  # noqa: BLE001
                    return self._json({"ok": False, "error": str(exc)}, 400)
                START.update(raw=raw, label=label)
                body = {"image": crop_image(raw)}
            if "clear_image" in body:  # forget it (memory only): later restarts start from text again
                START.update(raw=None, label="")
                return self._json({"ok": True})
            LOOP.call_soon_threadsafe(CMDS.put_nowait, body)
            self._json({"ok": True})
        elif self.path == "/server":
            args = server_args(body)
            LOOP.call_soon_threadsafe(CMDS.put_nowait, {"server": args})
            self._json({"ok": True, "args": args})
        elif self.path == "/local":
            for k in ("upscale", "sharp", "fps", "buffer"):
                if k in body:
                    LOCAL[k] = body[k] if k == "fps" else float(body[k]) if k == "sharp" else int(body[k])
            self._json({"ok": True})
        else:
            self.send_error(404)


async def main():
    global CMDS, LOOP
    CMDS, LOOP = asyncio.Queue(), asyncio.get_running_loop()
    tunnel = open_tunnel()
    srv = ThreadingHTTPServer((a.host, a.web_port), Web)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    log(f"web UI: http://127.0.0.1:{a.web_port}")
    if a.host == "0.0.0.0":
        import socket
        try:
            sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sk.connect(("10.255.255.255", 1))  # no packet sent: picks the LAN interface
            log(f"on your network (phone): http://{sk.getsockname()[0]}:{a.web_port}")
        except OSError:
            pass
    try:
        await asyncio.gather(session_loop(), player())
    finally:
        STOP.set()
        SINKS.close()
        if tunnel:
            tunnel.terminate()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
