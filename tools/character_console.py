"""Console for the live character mode, on this PC: start / stop a run and watch every stage of the pipeline live.

  cloud/facefix/.venv/bin/python tools/character_console.py        -> http://127.0.0.1:8780

Shows, left to right, what each stage outputs (the source as sent to JoyAI, JoyAI's output, the face pass = what is
played) with its rate and latency, the audio path (phone mic -> bridge -> voice stage -> delay line -> phone) with a
level meter and the backlog at each step, a latency breakdown, the phone's own playback report and the run's log.
Runs tools/character_live.py (any driver; phonecam = your phone is the camera + mic). Reads the bridge dashboard
(127.0.0.1:8765) and the phone page's /stats; only listens on 127.0.0.1.
"""
import json
import os
import re
import signal
import ssl
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / "cloud/facefix/.venv/bin/python")
LOG = REPO / ".tmp" / "console" / "run.log"
PIDF = LOG.with_name("run.json")     # the run survives a console restart: re-attached from here
DASH = "http://127.0.0.1:8765"
PORT = int(os.environ.get("CONSOLE_PORT", 8780))
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname, SSL_CTX.verify_mode = False, ssl.CERT_NONE

run = {"proc": None, "pid": None, "args": None, "t0": None, "ended": None, "rc": None}
lock = threading.Lock()


def characters():
    return sorted(p.parent.name for p in (REPO / "characters").glob("*/character.json"))


def get_json(url, timeout=0.8):
    try:
        with urllib.request.urlopen(url, timeout=timeout, context=SSL_CTX if url.startswith("https") else None) as r:
            return json.loads(r.read())
    except Exception:  # noqa: BLE001
        return None


def log_tail(n=60):
    try:
        lines = LOG.read_text(errors="replace").splitlines()
    except OSError:
        return []
    keep = [ln for ln in lines if "libwebrtc" not in ln and not ln.startswith("[bridge] played")]
    return keep[-n:]


def phone_url():
    try:
        m = re.findall(r"https://[\d.]+:\d+/\?t=[\w-]+", LOG.read_text(errors="replace"))
        return m[-1] if m else None
    except OSError:
        return None


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return Path(f"/proc/{pid}/cmdline").read_bytes().find(b"character_live.py") >= 0
    except (OSError, ProcessLookupError):
        return False


def adopt():
    """A run started by an earlier console process (still going): attach to it by PID."""
    try:
        j = json.loads(PIDF.read_text())
    except (OSError, ValueError):
        return
    if pid_alive(j["pid"]):
        run.update(pid=j["pid"], args=j["args"], t0=j["t0"], proc=None, ended=None, rc=None)


def is_alive():
    p = run.get("proc")
    if p is not None:
        return p.poll() is None
    return bool(run.get("pid")) and pid_alive(run["pid"])


def state():
    p = run["proc"]
    alive = is_alive()
    if (p is not None or run.get("pid")) and not alive and run["ended"] is None:
        run["ended"], run["rc"] = time.time(), (p.returncode if p is not None else None)
    url = phone_url() if alive else None
    st = {"running": alive, "args": run["args"], "t0": run["t0"], "ended": run["ended"], "rc": run["rc"],
          "characters": characters(), "phone_url": url, "log": log_tail(),
          "monitor_url": (REPO / ".tmp" / "monitor_url").read_text().strip() if (REPO / ".tmp" / "monitor_url").exists() else None}
    if alive:
        st["bridge"] = get_json(DASH + "/status")
        try:
            txt = LOG.read_text(errors="replace")
        except OSError:
            txt = ""
        drv = (run["args"] or {}).get("driver")
        if "Traceback" in txt:
            st["phase"] = ("bad", "error: see the log")
        elif st["bridge"] and st["bridge"].get("played"):
            st["phase"] = ("ok", "live")
        elif st["bridge"]:
            st["phase"] = ("wait", "connected: JoyAI is starting (first frames in a few seconds)")
        elif "[phone] connected" in txt or (drv == "webcam" and "voice stage ready" in txt):
            st["phase"] = ("wait", "your camera is coming in: opening the JoyAI session")
        elif url:
            st["phase"] = ("wait", "waiting for your phone: scan the QR code or open the link, then tap Start on the "
                                   "phone" if drv != "webcam" else "waiting for the webcam")
        else:
            st["phase"] = ("wait", "starting (voice stage + face pass load in ~20 s)")
        if url:
            base, tok = url.split("/?t=")
            st["phone"] = get_json(f"{base}/stats?t={tok}")
    return st


def preset(name):
    c = json.loads((REPO / "characters" / name / "character.json").read_text())
    ap = c["appearance"]
    return {"character": name, "prompt": (REPO / ap["prompt_file"]).read_text().strip(), "ref": ap["reference"],
            "prompt_file": ap["prompt_file"], "lpfast_preset": c["face_pass"]["lpfast_preset"],
            "voice": c.get("voice", {}).get("hf") or c.get("voice", {}).get("model") or c.get("voice", {}).get("reference_clip") or "none (your own voice)"}


def data_url_to_file(d, path):
    """Decode an uploaded image (data URL) and save it as PNG."""
    import base64
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(base64.b64decode(d.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not an image")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    return path


def bridge_cmd(cmd):
    req = urllib.request.Request(DASH + "/cmd", json.dumps(cmd).encode(), {"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=3).read()
        return {"ok": True}
    except Exception as e:  # noqa: BLE001
        return {"error": f"bridge not reachable ({e.__class__.__name__}): is a run live?"}


def apply_look(a):
    """Prompt and/or reference photo, live (one JoyAI session restart, ~1-2 s, memory reset)."""
    cmd = {}
    if a.get("prompt"):
        cmd["prompt"] = a["prompt"]
    r = a.get("ref")
    if r:
        cmd["ref"] = r
        if r.startswith("data:"):
            cmd["ref_name"] = a.get("ref_name") or "upload"
    return bridge_cmd(cmd) if cmd else {"error": "nothing to apply"}


def save_look(a):
    """Into the character's preset: prompt -> characters/<name>/prompt.txt (character.json points at it), photo ->
    characters/<name>/reference.png (git keeps the previous version)."""
    name = a.get("character")
    if name not in characters():
        return {"error": f"no character {name}"}
    d = REPO / "characters" / name
    cj = d / "character.json"
    c = json.loads(cj.read_text())
    saved = []
    if a.get("prompt"):
        (d / "prompt.txt").write_text(a["prompt"].strip() + "\n")
        c["appearance"]["prompt_file"] = f"characters/{name}/prompt.txt"
        saved.append("prompt")
    r = a.get("ref")
    if r and r != c["appearance"]["reference"]:
        dst = d / "reference.png"
        if r.startswith("data:"):
            data_url_to_file(r, dst)
        else:
            import shutil
            shutil.copyfile(REPO / r, dst)
        c["appearance"]["reference"] = f"characters/{name}/reference.png"
        saved.append("photo")
    cj.write_text(json.dumps(c, indent=2) + "\n")
    return {"ok": True, "saved": saved}


def restart(a):
    """Switch driver (or anything else fixed at start): stop, then start with the new settings."""
    stop()

    def later():
        t_end = time.time() + 100
        while time.time() < t_end and is_alive():
            time.sleep(0.5)
        start(a)
    threading.Thread(target=later, daemon=True).start()
    return {"ok": True}


def start(a):
    with lock:
        if is_alive():
            return {"error": "already running"}
        ch = a.get("character", "karen")
        if ch not in characters():
            return {"error": f"no character {ch}"}
        cmd = [PY, str(REPO / "tools/character_live.py"), ch, "--driver", a.get("driver", "phonecam"),
               "--max-seconds", str(int(float(a.get("minutes", 30)) * 60))]
        if a.get("driver") == "nuance":
            cmd.append("--phone")
        if str(a.get("pitch", "")).strip():
            cmd += ["--pitch", str(int(a["pitch"]))]
        pr = preset(ch)
        if a.get("prompt") and a["prompt"].strip() != pr["prompt"]:
            cmd += ["--prompt", a["prompt"].strip()]
        r = a.get("ref")
        if r and r != pr["ref"]:
            p = data_url_to_file(r, LOG.with_name("ref_upload.png")) if r.startswith("data:") else REPO / r
            cmd += ["--ref", str(p)]
        LOG.parent.mkdir(parents=True, exist_ok=True)
        f = open(LOG, "w")
        run.update(proc=subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True),
                   args=a, t0=time.time(), ended=None, rc=None)
        run["pid"] = run["proc"].pid
        PIDF.write_text(json.dumps({"pid": run["pid"], "args": a, "t0": run["t0"]}))
        return {"ok": True}


def stop():
    if not is_alive():
        return {"ok": True}
    pid, p = run["pid"], run["proc"]
    os.kill(pid, signal.SIGTERM)           # character_live cleans up (bridge, phone page, sinks)

    def reaper():
        t_end = time.time() + 90
        while time.time() < t_end and pid_alive(pid):
            if p is not None:
                p.poll()
            time.sleep(0.5)
        if pid_alive(pid):
            os.killpg(pid, signal.SIGKILL)
    threading.Thread(target=reaper, daemon=True).start()
    return {"ok": True}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path.startswith("/api/state"):
            return self._send(200, json.dumps(state()))
        if self.path.startswith("/api/preset?c="):
            try:
                return self._send(200, json.dumps(preset(self.path.split("=", 1)[1])))
            except (OSError, KeyError, ValueError) as e:
                return self._send(404, json.dumps({"error": str(e)}))
        if self.path.startswith("/img?p="):    # previews: character photos and assets only
            from urllib.parse import unquote
            p = (REPO / unquote(self.path.split("=", 1)[1].split("&")[0])).resolve()
            if (REPO / "characters" in p.parents or REPO / "assets" in p.parents) and p.is_file():
                ct = {".png": "image/png", ".webp": "image/webp"}.get(p.suffix.lower(), "image/jpeg")
                return self._send(200, p.read_bytes(), ct)
        if self.path == "/api/voices":
            cat = json.loads((REPO / "voices" / "catalog.json").read_text())
            return self._send(200, json.dumps({k: v.get("label", k) for k, v in cat.items() if not k.startswith("_")}))
        if self.path == "/api/assets":
            imgs = sorted(str(p.relative_to(REPO)) for d in ("characters", "assets") for p in (REPO / d).rglob("*")
                          if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp") and p.stat().st_size < 20e6)
            return self._send(200, json.dumps(imgs))
        self._send(404, "{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            a = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            a = {}
        if self.path == "/api/start":
            return self._send(200, json.dumps(start(a)))
        if self.path == "/api/stop":
            return self._send(200, json.dumps(stop()))
        if self.path == "/api/restart":
            return self._send(200, json.dumps(restart(a)))
        if self.path == "/api/apply":
            return self._send(200, json.dumps(apply_look(a)))
        if self.path == "/api/bypass":    # face pass bypass: JoyAI output goes straight out
            return self._send(200, json.dumps(bridge_cmd({"facefix": not a.get("on")})))
        if self.path == "/api/voice":     # live voice switch (voices/catalog.json)
            return self._send(200, json.dumps(bridge_cmd({"voice_select": a.get("name", "")})))
        if self.path == "/api/avchain":
            return self._send(200, json.dumps(bridge_cmd({"av_chain": float(a.get("ms", 0))})))
        if self.path == "/api/reset":
            return self._send(200, json.dumps(bridge_cmd({"reset": True})))
        if self.path == "/api/save":
            try:
                return self._send(200, json.dumps(save_look(a)))
            except Exception as e:  # noqa: BLE001
                return self._send(200, json.dumps({"error": str(e)}))
        self._send(404, "{}")


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Character Console</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0e1013;--card:#171a1f;--line:#2a2f37;--fg:#e6e8eb;--dim:#8b939e;--ok:#3fb27f;--warn:#e0a03a;--bad:#e05a4f;--acc:#5a9cf0}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,-apple-system,sans-serif}
header{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--bg);z-index:2}
h1{font-size:15px;margin:0 12px 0 0;font-weight:600}
select,input{background:#0b0d10;color:var(--fg);border:1px solid var(--line);border-radius:5px;padding:5px 7px;font:inherit}
button{border:0;border-radius:5px;padding:6px 14px;font:600 13px system-ui;color:#fff;cursor:pointer}
#start{background:var(--ok)} #stop{background:var(--bad)} button:disabled{opacity:.4;cursor:default}
.pill{padding:3px 9px;border-radius:99px;font-size:12px;background:#222;color:var(--dim)} .pill.on{background:#123d2a;color:#7fe0b0}
#link{font:13px ui-monospace,monospace;color:var(--acc);user-select:all;word-break:break-all}
#copy{background:var(--acc)} .b2{background:var(--acc)} .b3{background:#3a414c} #switch{background:var(--warn);color:#111}
.btn2{display:block;text-align:center;background:#3a414c;border-radius:5px;padding:5px;cursor:pointer;font-weight:600;font-size:12px} #phase{font-size:13px} #phase.wait{color:var(--warn)} #phase.ok{color:var(--ok)} #phase.bad{color:var(--bad)}
#phonebox{display:flex;gap:16px;align-items:center;margin:12px 16px 0;padding:12px;background:var(--card);border:1px solid var(--line);border-radius:8px}
#qr{background:#fff;padding:8px;border-radius:6px;line-height:0} #qr img,#qr canvas{width:96px;height:96px}
main{padding:14px 16px;display:grid;gap:14px}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;align-items:stretch}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;min-width:0;position:relative}
.card h2{font-size:12px;font-weight:600;margin:0 0 6px;color:var(--dim);text-transform:uppercase;letter-spacing:.05em}
.card img{width:100%;aspect-ratio:480/840;object-fit:contain;background:#000;border-radius:4px;display:block}
.kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;margin-top:6px;font-variant-numeric:tabular-nums}
.kv span:nth-child(odd){color:var(--dim)} .kv span:nth-child(even){text-align:right}
.step{position:absolute;top:10px;right:10px;font-size:11px;color:var(--dim)}
.meter{height:8px;background:#0b0d10;border-radius:4px;overflow:hidden;margin:4px 0 2px}
.meter i{display:block;height:100%;background:linear-gradient(90deg,var(--ok),var(--warn) 80%,var(--bad));width:0;transition:width .25s}
.bar{display:flex;height:26px;border-radius:5px;overflow:hidden;background:#0b0d10;margin-top:6px}
.bar div{display:flex;align-items:center;justify-content:center;font-size:11px;white-space:nowrap;overflow:hidden;color:#0b0d10;font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;color:var(--dim);font-size:12px}
.legend b{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
pre{margin:0;max-height:240px;overflow:auto;font:11px/1.45 ui-monospace,monospace;color:#c8ccd2;white-space:pre-wrap}
.warn{color:var(--warn)} .bad{color:var(--bad)} .ok{color:var(--ok)}
</style></head><body>
<header><h1>Character Console</h1>
<select id="ch"></select>
<select id="drv"><option value="phonecam">you: phone (camera + mic)</option><option value="webcam">you: laptop webcam + mic</option><option value="nuance">Nuance persona (credits)</option></select>
<label>min <input id="min" type="number" value="30" min="1" max="120" style="width:58px"></label>
<label>pitch <input id="pitch" placeholder="auto" style="width:52px"></label>
<button id="start">Start</button><button id="stop" disabled>Stop</button><button id="switch" hidden>Switch driver (restart)</button>
<span id="pill" class="pill">stopped</span><span id="phase"></span>
<span id="mon" style="margin-left:auto;font-size:12px;color:var(--dim)"></span></header>
<section id="phonebox" hidden><div id="qr"></div><div style="min-width:0"><div style="color:var(--dim);margin-bottom:4px">Open on your phone (same Wi-Fi) — or scan the code:</div>
<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><code id="link"></code><button id="copy">Copy</button></div>
<div style="color:var(--dim);margin-top:6px;font-size:12px">First time: the phone warns about the certificate — Show Details → visit this website. Then tap Start on the phone.</div></div></section>
<main>
<div class="card" id="look"><h2>Look — JoyAI reference photo + prompt</h2>
 <div style="display:grid;grid-template-columns:150px 1fr;gap:14px;align-items:start">
  <div><img id="refimg" alt="" style="width:150px;aspect-ratio:3/4;object-fit:cover;border-radius:6px;background:#000;display:block">
   <label class="btn2" style="margin-top:6px">Upload photo…<input id="file" type="file" accept="image/*" hidden></label>
   <select id="lib" style="width:150px;margin-top:6px"><option value="">…or pick from library</option></select></div>
  <div style="min-width:0"><textarea id="prompt" rows="4" style="width:100%;resize:vertical;font:13px system-ui"></textarea>
   <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;align-items:center">
    <button id="apply" class="b2">Apply live</button><button id="reset" class="b2" title="Resets JoyAI memory and recalibrates the face pass: look at the camera with a neutral face">Reset memory + neutral</button>
    <label title="Switch the voice live: the old voice unloads, the new one loads (~10-25 s, muted meanwhile)">Voice
     <select id="voicesel"><option value="">…</option></select></label><span id="voicest" style="color:var(--dim);font-size:12px"></span>
    <button id="bypass" class="b3" title="Skip the face pass: the JoyAI output is sent out directly">Bypass face pass: off</button>
    <button id="revert" class="b3">Revert to preset</button><button id="save" class="b3">Save to preset</button>
    <span id="lookmsg" style="color:var(--dim)"></span></div>
   <div id="livenow" style="color:var(--dim);font-size:12px;margin-top:8px"></div></div>
 </div></div>
<div class="row">
 <div class="card"><span class="step">1</span><h2>Source (as sent to JoyAI)</h2><img id="v_input" alt=""><div class="kv" id="k_in"></div></div>
 <div class="card"><span class="step">2</span><h2>JoyAI output</h2><img id="v_joyai" alt=""><div class="kv" id="k_joy"></div></div>
 <div class="card"><span class="step">3</span><h2>Face pass = played</h2><img id="v_final" alt=""><div class="kv" id="k_fin"></div></div>
 <div class="card"><span class="step">4</span><h2>Phone</h2><div class="kv" id="k_ph"></div>
   <div style="margin-top:10px;display:flex;gap:8px;align-items:center" title="Shifts the character's voice against the picture (all outputs: window, OBS, phone). Lips move before you hear them = press −. Saved for this camera input.">
    <b>Voice timing</b><button id="av_m" class="b3">−50 ms</button><span id="av_v" style="min-width:64px;text-align:center">–</span><button id="av_p" class="b3">+50 ms</button>
    <span style="color:var(--dim);font-size:12px">voice late → −, early → +</span></div>
   <h2 style="margin-top:14px">Latency (video, capture → screen)</h2><div class="bar" id="lat"></div><div class="legend" id="latl"></div></div>
</div>
<div class="card"><h2>Audio path</h2><div class="row" id="audio"></div></div>
<div class="card"><h2>Log</h2><pre id="log"></pre></div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"></script>
<script>
let qrFor=null;
document.addEventListener("click",e=>{ if(e.target.id!=="copy")return; const u=$("link").textContent;
  (navigator.clipboard?navigator.clipboard.writeText(u):Promise.reject()).then(()=>{e.target.textContent="Copied";setTimeout(()=>e.target.textContent="Copy",1500);})
  .catch(()=>{const r=document.createRange(); r.selectNodeContents($("link")); const s=getSelection(); s.removeAllRanges(); s.addRange(r);});});
const $=id=>document.getElementById(id); let wasRunning=null, dashUp=false;
const fmt=(v,u="",d=0)=>v==null||Number.isNaN(v)?"–":(+v).toFixed(d)+u;
function kv(el,rows){el.innerHTML=rows.map(([k,v,c])=>`<span>${k}</span><span class="${c||""}">${v}</span>`).join("");}
function meter(db){return Math.max(0,Math.min(100,(db+60)/60*100));}
function views(on){for(const k of ["input","joyai","final"]){const im=$("v_"+k); im.src=on?`http://127.0.0.1:8765/stream/${k}?fps=10&t=${Date.now()}`:"";}}
async function post(p,b){await fetch(p,{method:"POST",body:JSON.stringify(b||{})});}
// ---- look: the editors hold the photo + prompt; the preset is what the character folder has
let look={ref:null, refName:null, prompt:""}, pre=null, running=false, runArgs=null;
const startArgs=()=>({character:$("ch").value,driver:$("drv").value,minutes:$("min").value,pitch:$("pitch").value,prompt:$("prompt").value,ref:look.ref});
$("start").onclick=()=>post("/api/start",startArgs());
$("switch").onclick=()=>{post("/api/restart",startArgs()); $("switch").hidden=true; msg("restarting with the new driver…");};
function msg(m,c){$("lookmsg").textContent=m; $("lookmsg").style.color=c||"var(--dim)";}
function showRef(){ $("refimg").src=look.ref&&look.ref.startsWith("data:")?look.ref:(look.ref?"/img?p="+encodeURIComponent(look.ref):""); }
function dirty(){ return pre && ($("prompt").value.trim()!==pre.prompt || look.ref!==pre.ref); }
async function loadPreset(name, applyIfLive){
  pre=await (await fetch("/api/preset?c="+encodeURIComponent(name))).json();
  look={ref:pre.ref, refName:pre.ref, prompt:pre.prompt}; $("prompt").value=pre.prompt; showRef(); msg("");
  if(applyIfLive&&running){ const r=await (await fetch("/api/apply",{method:"POST",body:JSON.stringify({prompt:pre.prompt,ref:pre.ref})})).json();
    msg(r.error||(`${name}'s photo + prompt applied live`+(runArgs&&runArgs.character!==name?` — voice + face-pass preset change on restart`:"")), r.error?"var(--bad)":"var(--ok)"); }
}
$("ch").onchange=()=>loadPreset($("ch").value,true);
$("file").onchange=e=>{const f=e.target.files[0]; if(!f)return; const rd=new FileReader();
  rd.onload=()=>{look.ref=rd.result; look.refName=f.name; showRef(); msg(`photo "${f.name}" loaded — Apply live / Save to preset`);}; rd.readAsDataURL(f);};
$("lib").onchange=e=>{ if(!e.target.value)return; look.ref=e.target.value; look.refName=e.target.value; showRef(); msg("photo picked — Apply live / Save to preset"); e.target.value=""; };
$("apply").onclick=async()=>{ const r=await (await fetch("/api/apply",{method:"POST",body:JSON.stringify({prompt:$("prompt").value,ref:look.ref,ref_name:look.refName})})).json();
  msg(r.error||"applied live (JoyAI restarts its session: ~2 s, memory reset)", r.error?"var(--bad)":"var(--ok)"); };
let avChain=null;
async function avStep(d){ if(avChain==null) return; const r=await (await fetch("/api/avchain",{method:"POST",body:JSON.stringify({ms:avChain+d})})).json();
  if(r.error) msg(r.error,"var(--bad)"); else { avChain+=d; $("av_v").textContent=(avChain>0?"+":"")+avChain+" ms"; } }
$("av_m").onclick=()=>avStep(-50); $("av_p").onclick=()=>avStep(50);
(async()=>{ try{ const v=await (await fetch("/api/voices")).json();
  $("voicesel").innerHTML='<option value="">— pick to switch —</option>'+Object.entries(v).map(([k,l])=>`<option value="${k}">${l}</option>`).join(""); }catch(_){} })();
$("voicesel").onchange=async()=>{ const n=$("voicesel").value; if(!n) return;
  const r=await (await fetch("/api/voice",{method:"POST",body:JSON.stringify({name:n})})).json();
  msg(r.error||"switching voice… (muted while it loads)", r.error?"var(--bad)":"var(--ok)"); $("voicesel").value=""; };
let bypassOn=false;
$("bypass").onclick=async()=>{ const r=await (await fetch("/api/bypass",{method:"POST",body:JSON.stringify({on:!bypassOn})})).json();
  if(r.error) msg(r.error,"var(--bad)"); else { bypassOn=!bypassOn; showBypass(); } };
function showBypass(){ $("bypass").textContent="Bypass face pass: "+(bypassOn?"ON":"off"); $("bypass").style.background=bypassOn?"var(--bad)":""; }
$("reset").onclick=async()=>{ const r=await (await fetch("/api/reset",{method:"POST",body:"{}"})).json(); msg(r.error||"memory reset + face pass recalibrated (look at the camera, neutral face)", r.error?"var(--bad)":"var(--ok)"); };
$("revert").onclick=()=>loadPreset($("ch").value,false);
$("save").onclick=async()=>{ const r=await (await fetch("/api/save",{method:"POST",body:JSON.stringify({character:$("ch").value,prompt:$("prompt").value,ref:look.ref})})).json();
  if(r.error) return msg(r.error,"var(--bad)"); await loadPreset($("ch").value,false); msg(`saved to ${$("ch").value}'s preset: ${r.saved.join(" + ")||"nothing changed"}`,"var(--ok)"); };
$("drv").onchange=()=>{ $("switch").hidden=!(running&&runArgs&&runArgs.driver!==$("drv").value); };
fetch("/api/assets").then(r=>r.json()).then(l=>{$("lib").innerHTML+=l.map(p=>`<option value="${p}">${p.replace(/^(characters|assets)\//,"")}</option>`).join("");});
$("stop").onclick=()=>post("/api/stop");
async function tick(){
  let s; try{s=await (await fetch("/api/state")).json();}catch(e){return;}
  if(!$("ch").options.length){ $("ch").innerHTML=s.characters.map(c=>`<option>${c}</option>`).join("");
    if(s.running&&s.args){ $("ch").value=s.args.character; $("drv").value=s.args.driver; } loadPreset($("ch").value,false); }
  running=s.running; runArgs=s.args; $("switch").hidden=!(running&&runArgs&&runArgs.driver!==$("drv").value);
  const bb=s.bridge; $("livenow").textContent=bb?`live now: photo ${String(bb.ref||"–").split("/").slice(-2).join("/")} · prompt "${(bb.prompt||"").slice(0,90)}${(bb.prompt||"").length>90?"…":""}"`:"(not live: Apply live works once a run is live; Start uses the photo + prompt above)";
  if(dirty()&&!$("lookmsg").textContent) msg("unsaved changes","var(--warn)");
  $("start").disabled=s.running; $("stop").disabled=!s.running;
  $("pill").textContent=s.running?`running ${Math.round((Date.now()/1000-s.t0)/60)} min · ${s.args.driver}`:(s.ended?`stopped (exit ${s.rc})`:"stopped");
  $("pill").className="pill"+(s.running?" on":"");
  if(s.monitor_url&&!$("mon").dataset.u){ $("mon").dataset.u=s.monitor_url;
    $("mon").innerHTML=`monitor (2nd phone, home Wi-Fi): <code style="color:var(--acc);user-select:all">${s.monitor_url}</code> <span id="qrm" style="display:inline-block;vertical-align:middle;background:#fff;padding:3px;border-radius:4px;line-height:0"></span>`;
    if(window.QRCode) new QRCode($("qrm"),{text:s.monitor_url,width:64,height:64,correctLevel:QRCode.CorrectLevel.L}); }
  const showPhone=!!(s.running&&s.phone_url); $("phonebox").hidden=!showPhone;
  if(showPhone&&qrFor!==s.phone_url){ $("link").textContent=s.phone_url; $("qr").innerHTML="";
    if(window.QRCode) new QRCode($("qr"),{text:s.phone_url,width:96,height:96,correctLevel:QRCode.CorrectLevel.M}); qrFor=s.phone_url; }
  const ph_=s.phase||(s.running?["wait","starting"]:null); $("phase").textContent=ph_?ph_[1]:""; $("phase").className=ph_?ph_[0]:"";
  if(s.bridge&&s.bridge.sync&&s.bridge.sync.chain_ms!=null){ avChain=s.bridge.sync.chain_ms; $("av_v").textContent=(avChain>0?"+":"")+avChain+" ms"; }
  if(s.bridge){ const b=s.bridge; $("voicest").textContent=b.voice_switching?`loading ${b.voice_switching}…`:
     b.voice_switch_failed?`switch failed: ${b.voice_switch_failed}`:`now: ${b.voice_name||"–"}`;
     $("voicest").style.color=b.voice_switch_failed?"var(--bad)":b.voice_switching?"var(--warn,orange)":"var(--dim)"; }
  if(s.bridge&&s.bridge.facefix_available){ bypassOn=!s.bridge.facefix_on; showBypass(); }
  const b=s.bridge, p=s.phone, sy=b&&b.sync||{}, lv=sy.levels||{}, fl=sy.flow||{}, vo=sy.voice||{}, mo=sy.monitor||{};
  const up=!!b; if(up!==dashUp){views(up); dashUp=up;}
  const r=(p&&p.rates)||{}, ph=(p&&p.phone)||{};
  kv($("k_in"),[["phone camera → laptop",fmt(r.cam_in," fps",1)],["bridge input",fmt(b&&b.input_fps," fps",1)],["sending to JoyAI",fmt(b&&b.send_fps," fps",1),b&&b.send_fps<20?"warn":""],
    ["dropped as stale (>120 ms)",b?b.input_stale:"–"],["camera arrival gaps p50/p90/max",p&&p.cam_arrival?`${p.cam_arrival.gap_p50_ms}/${p.cam_arrival.gap_p90_ms}/${p.cam_arrival.gap_max_ms} ms`:"–"]]);
  kv($("k_joy"),[["round trip",fmt(b&&b.joyai_rt_ms," ms"),b&&b.joyai_rt_ms>900?"warn":""],["JoyAI output",fmt(b&&b.server_fps," fps",1)],["playout queue",b?`${b.queue} / buffer ${b.buffer}`:"–"],["dropped / repeated",b?`${b.dropped} / ${b.repeated}`:"–"]]);
  const hit=b&&b.facefix_hit?b.facefix_hit.split("/"):null, hr=hit?hit[0]/hit[1]*100:null;
  kv($("k_fin"),[["face pass render",fmt(b&&b.facefix_ms," ms")],["refined",fmt(hr,"%",1),hr!=null&&hr<97?"warn":""],["played",b?b.played:"–"],["glass to glass",fmt(b&&b.g2g_ms," ms")]]);
  const fresh=ph.t&&(Date.now()/1000-ph.t<6);
  kv($("k_ph"),[["connected",p?p.clients:"–",p&&p.clients?"ok":"warn"],["video to phone",fmt(r.video_out," fps",1)],["shown on phone",fresh?fmt(ph.shown_fps," fps",1):"–"],
     ["phone audio buffer",fresh?fmt(ph.buf_ms," ms"):"–"],["phone output latency",fresh?fmt(ph.out_lat_ms," ms"):"–"],["sync slider",fresh?fmt(ph.offset_ms," ms"):"–"]]);
  // latency bar: JoyAI round trip | playout + face pass | phone buffer + output
  const rt=b&&b.joyai_rt_ms, g=b&&b.g2g_ms, phb=fresh?(ph.buf_ms||0)+(ph.out_lat_ms||0):null;
  const segs=[["JoyAI round trip (network + GPU)",rt,"#5a9cf0"],["playout buffer + face pass",g!=null&&rt!=null?Math.max(0,g-rt):null,"#b78cf0"],["phone (audio-led buffer + output)",phb,"#3fb27f"]];
  const tot=segs.reduce((a,x)=>a+(x[1]||0),0)||1;
  $("lat").innerHTML=segs.map(([n,v,c])=>v?`<div style="width:${v/tot*100}%;background:${c}" title="${n}">${Math.round(v)}</div>`:"").join("");
  $("latl").innerHTML=segs.map(([n,v,c])=>`<span><b style="background:${c}"></b>${n}: ${fmt(v," ms")}</span>`).join("")+`<span>total ≈ ${fmt(tot>1?tot:null," ms")} (+ camera → laptop, not measured)</span>`;
  const A=[["1 · phone mic",p&&p.mic_db,`${fmt(r.mic_in,"/s",0)} chunks`],
           ["2 · bridge mic in",lv.in,"from the phone (PulseAudio)"],
           ["3 · voice stage out",lv.voice,`algorithmic ${fmt(vo.delay_ms," ms")} · inside ${fmt(vo.in_stage_ms," ms")}`+(fl.stall_ms>300?` · <span class="bad">behind ${fl.stall_ms} ms</span>`:"")],
           ["4 · delay line",null,`target ${fmt(sy.delay_ms," ms")} · queued ${fmt(fl.buf!=null?fl.buf*10:null," ms")} · dropped ${mo.dropped??"–"} · inserted ${mo.inserted??"–"}`],
           ["5 · to the phone",lv.out,`${fmt(r.audio_out,"/s",0)} chunks sent`],
           ["6 · phone playback",null,fresh?`buffer ${fmt(ph.buf_ms," ms")} · output ${fmt(ph.out_lat_ms," ms")}`:"–"]];
  $("audio").innerHTML=A.map(([n,db,info])=>`<div><div style="color:var(--dim)">${n}</div>${db!==null&&db!==undefined?`<div class="meter"><i style="width:${meter(db)}%"></i></div><div>${fmt(db," dBFS",0)}</div>`:'<div class="meter" style="opacity:.25"></div><div>&nbsp;</div>'}<div style="color:var(--dim);font-size:12px">${info}</div></div>`).join("");
  const lg=$("log"), atBottom=lg.scrollTop+lg.clientHeight>=lg.scrollHeight-8;
  lg.innerHTML=s.log.map(l=>{const e=l.replace(/&/g,"&amp;").replace(/</g,"&lt;");return /behind|error|Traceback|lost|fail/i.test(l)?`<span class="warn">${e}</span>`:e;}).join("\n");
  if(atBottom) lg.scrollTop=lg.scrollHeight;
}
setInterval(tick,1000); tick();
</script></body></html>"""


def ensure_monitor():
    """The watch-only monitor (tools/monitor_view.py) for a second phone on the home Wi-Fi: its own low-priority
    process, independent of runs (it survives them); started here if it isn't running."""
    import socket
    s = socket.socket(); s.settimeout(0.5)
    try:
        if s.connect_ex(("192.168.78.142" if False else "127.0.0.1", 0)) == 0:
            pass
    finally:
        s.close()
    if subprocess.run(["ss", "-tln", "sport", "= :8450"], capture_output=True, text=True).stdout.count("LISTEN"):
        return
    subprocess.Popen([PY, str(REPO / "tools/monitor_view.py")], cwd=REPO, stdout=open(LOG.with_name("monitor.log"), "w"),
                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)


def main():
    adopt()
    ensure_monitor()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
    print(f"[console] http://127.0.0.1:{PORT}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # a run outlives the console (Stop button stops it)
    main()
