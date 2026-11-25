"""Live character mode: a conversational AI persona (Nuance Live) becomes one of OUR characters, live:

    persona video -> SRT -> live bridge -> JoyAI (cloud, the character's appearance) -> lpfast face pass (laptop,
                     the character's preset) -> window / OBS
    persona audio -> the character's voice (RVC on the JoyAI GPU) -> delayed to match the video -> headphones
    your mic      -> the persona (it answers you as the character: its personality + name)

A character is a folder characters/<name>/ (see characters/README.md): character.json + reference photo +
personality; the voice model lives in models/ (not in git). Other pipelines are untouched (new files only; the
bridge's options used here all default to off).

  tools/character_live.py karen                 live: talk to it (HEADPHONES), output window + dashboard :8765
  tools/character_live.py karen --test          the character's recorded persona clip instead of a live session
                                                (no Nuance credits), nothing audible, records the result
  tools/character_live.py karen --test --seconds 30 --out ~/Videos/ai-livestream-tests/x.mp4
  options: --voice-host '' (voice on the laptop GPU: starves lpfast, see tools/rvc_live/README.md), --mic SOURCE,
           --max-seconds N (live session length), --no-face-pass, --bridge-args '...' (extra live_bridge flags)
Needs: JoyAI up + tunnel (cloud/infra/tunnel.sh start) and the voice installed there (cloud/infra/voice_up.sh joyai).
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / "cloud/facefix/.venv/bin/python")
NPY = str(REPO / "tools/nuance_live/.venv/bin/python")
PORT = 9024


def load_character(name):
    d = REPO / "characters" / name
    f = d / "character.json"
    if not f.exists():
        avail = sorted(p.name for p in (REPO / "characters").iterdir() if (p / "character.json").exists())
        sys.exit(f"no character '{name}' (characters/{name}/character.json); available: {', '.join(avail)}")
    c = json.loads(f.read_text())
    for path in (c["appearance"]["reference"], c["appearance"]["prompt_file"], c["voice"]["model"],
                 c["conversation"]["personality_file"]):
        if not (REPO / path).exists():
            sys.exit(f"{name}: missing {path}")
    return c


def sink(name):
    return subprocess.check_output(["pactl", "load-module", "module-null-sink", f"sink_name={name}",
                                    f"sink_properties=device.description={name}"]).decode().strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("character")
    ap.add_argument("--test", action="store_true", help="drive from the recorded persona clip (no credits, silent)")
    ap.add_argument("--clip", default=None, help="--test: another recorded clip (default: the character's test_clip)")
    ap.add_argument("--seconds", type=float, default=60, help="--test: clip seconds")
    ap.add_argument("--max-seconds", type=int, default=300, help="live: Nuance session length")
    ap.add_argument("--out", default=None, help="record what you see + hear (side by side with the driver) to this mp4")
    ap.add_argument("--voice-host", default="joyai", help="GPU machine for the voice ('' = this laptop)")
    ap.add_argument("--mic", default="default", help="live: PulseAudio source you talk to the persona with")
    ap.add_argument("--url", default="ws://127.0.0.1:18080/ws")
    ap.add_argument("--dashboard", type=int, default=8765)
    ap.add_argument("--no-face-pass", action="store_true")
    ap.add_argument("--no-voice", action="store_true", help="plain persona voice (no conversion)")
    ap.add_argument("--bridge-args", default="", help="extra live_bridge.py flags")
    a = ap.parse_args()
    c = load_character(a.character)
    v, conv = c["voice"], c["conversation"]
    tmp = REPO / ".tmp" / "character"
    tmp.mkdir(parents=True, exist_ok=True)
    out = a.out or (str(Path.home() / "Videos/ai-livestream-tests" / f"{a.character}-{time.strftime('%m%d-%H%M')}.mp4")
                    if a.test else None)

    mods = [sink("char_audio")]
    env = dict(os.environ)
    if a.test:
        mods.append(sink("char_out"))
        env["PULSE_SINK"] = "char_out"
    run_s = a.seconds if a.test else a.max_seconds
    bridge = [PY, str(REPO / "cloud/client/live_bridge.py"), "--url", a.url, "--input", f"srt://0.0.0.0:{PORT}",
              "--portrait", "--ref", str(REPO / c["appearance"]["reference"]),
              "--prompt", (REPO / c["appearance"]["prompt_file"]).read_text().strip(),
              "--audio-monitor", "--monitor-source", "char_audio.monitor",
              "--voice", "off" if a.no_voice else v["kind"], "--voice-model", str(REPO / v["model"]), "--voice-pitch", str(v["pitch"]),
              "--voice-index-rate", str(v["index_rate"]), "--voice-block-ms", str(v["block_ms"]),
              "--seconds", str(int(run_s + 40))]
    if a.voice_host and not a.no_voice:
        bridge += ["--voice-host", a.voice_host]
    if not a.no_face_pass:
        bridge += ["--facefix-preset", c["face_pass"]["lpfast_preset"]]
    bridge += ["--output", str(tmp / "played.mp4")] if a.test else ["--output", "ffplay", "--dashboard", str(a.dashboard)]
    if out:
        bridge += ["--record-av", out, "--record-av-sbs"]
    bridge += shlex.split(a.bridge_args)
    log = open(tmp / "bridge.log", "w")
    procs = []
    try:
        print(f"[character] {c['name']}: starting the bridge (log {tmp / 'bridge.log'})", flush=True)
        br = subprocess.Popen(bridge, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append(br)
        t_end = time.time() + 120
        while time.time() < t_end and br.poll() is None:
            txt = (tmp / "bridge.log").read_text(errors="replace")
            if "audio monitor" in txt or "Traceback" in txt:
                break
            time.sleep(1)
        txt = (tmp / "bridge.log").read_text(errors="replace")
        for line in txt.splitlines():
            if "voice stage" in line or "Traceback" in line:
                print(line, flush=True)
        if br.poll() is not None or "Traceback" in txt:
            sys.exit("[character] the bridge did not start (see the log)")
        if a.test:
            clip = a.clip or str(REPO / conv["test_clip"])
            drv = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-re", "-i", clip, "-t", str(a.seconds), "-map", "0:v", "-c:v", "libx264",
                 "-preset", "veryfast", "-tune", "zerolatency", "-g", "30", "-b:v", "5M", "-f", "mpegts",
                 f"srt://127.0.0.1:{PORT}?mode=caller&latency=20", "-map", "0:a", "-ac", "1", "-f", "pulse",
                 "-device", "char_audio", "character-clip"])
        else:
            drv = subprocess.Popen(
                [NPY, str(REPO / "tools/nuance_live/nuance_source.py"), "--srt", f"srt://127.0.0.1:{PORT}",
                 "--sink", "char_audio", "--persona", conv["persona_id"],
                 "--prompt", (REPO / conv["personality_file"]).read_text().strip(), "--say", conv["first_line"],
                 "--expressiveness", str(conv.get("expressiveness", 1.0)), "--mic", a.mic,
                 "--max-seconds", str(a.max_seconds)])
        procs.append(drv)
        drv.wait()
        time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
                try:
                    p.wait(30)
                except subprocess.TimeoutExpired:
                    p.kill()
        for m in mods:
            subprocess.run(["pactl", "unload-module", m])
        txt = (tmp / "bridge.log").read_text(errors="replace")
        for line in txt.splitlines():
            if "A/V recording" in line or "RESULT" in line:
                print(line[:400])


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    main()
