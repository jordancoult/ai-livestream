"""Live character mode: a driver becomes one of OUR characters, live. The driver is either a conversational AI
persona (Nuance Live, default) or YOU (webcam or phone camera + your mic):

    driver video -> live bridge -> JoyAI (cloud, the character's appearance) -> lpfast face pass (laptop, the
                    character's preset) -> window / OBS
    driver audio -> the character's voice (RVC on the JoyAI GPU) -> delayed to match the video -> headphones
    --driver nuance: your mic -> the persona (it answers you as the character: its personality + name)

A character is a folder characters/<name>/ (see characters/README.md): character.json + reference photo +
personality; the voice model lives in models/ (not in git). Other pipelines are untouched (new files only; the
bridge's options used here all default to off).

  tools/character_live.py karen                 live: talk to it (HEADPHONES), output window + dashboard :8765
  tools/character_live.py karen --test          the character's recorded persona clip instead of a live session
                                                (no Nuance credits), nothing audible, records the result
  tools/character_live.py karen --test --seconds 30 --out test-outputs/x.mp4
  tools/character_live.py karen --driver webcam  YOU drive: webcam + mic -> Karen (HEADPHONES), until Ctrl+C
  tools/character_live.py karen --driver phone   YOU drive: iPhone camera over AirPlay (tools/iphone_airplay.sh is
                                                started for you; mirror the phone to "JoyAI-Cam") + laptop mic
  tools/character_live.py karen --driver srt     YOU drive: a phone app streaming SRT to this laptop (Larix, port
                                                9000; cloud/docs/iphone-camera.md) + laptop mic
  tools/character_live.py karen --phone          talk to it FROM YOUR PHONE (same Wi-Fi): a page shows the character
                                                and plays its voice; the phone's mic is your voice (tools/phone_view.py)
  tools/character_live.py karen --driver webcam --test   the self-drive test clip (a face + a male voice) instead
                                                of your camera/mic (same path as live from the bridge on)
  options: --mic SOURCE (your voice: any PulseAudio source, e.g. a headset mic), --camera /dev/videoN,
           --pitch N (override the voice pitch; self-drive uses voice.pitch_self), --voice-host '' (voice on the laptop GPU: starves lpfast, see tools/rvc_live/README.md), --mic SOURCE,
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
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = str(REPO / "cloud/facefix/.venv/bin/python")
NPY = str(REPO / "tools/nuance_live/.venv/bin/python")
PORT = 9024
CAM_PORT = 9025   # --phone: the phone camera -> the Nuance persona (it sees you)


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


SELF = ("webcam", "phone", "srt")


def sink(name):
    return subprocess.check_output(["pactl", "load-module", "module-null-sink", f"sink_name={name}",
                                    f"sink_properties=device.description={name}"]).decode().strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("character")
    ap.add_argument("--driver", default="nuance", choices=("nuance",) + SELF,
                    help="who drives the character: the Nuance persona, or you via webcam / phone (AirPlay) / srt app")
    ap.add_argument("--camera", default="/dev/video0", help="--driver webcam: v4l2 device")
    ap.add_argument("--srt-port", type=int, default=9000, help="--driver phone/srt: port the phone stream arrives on")
    ap.add_argument("--pitch", type=int, default=None, help="voice pitch override (semitones)")
    ap.add_argument("--phone", action="store_true", help="watch/hear/talk from a phone browser on the same Wi-Fi "
                    "(its mic replaces --mic; the character's voice plays there instead of the laptop)")
    ap.add_argument("--phone-port", type=int, default=8443)
    ap.add_argument("--test", action="store_true", help="drive from the recorded persona clip (no credits, silent)")
    ap.add_argument("--clip", default=None, help="--test: another recorded clip (default: the character's test_clip)")
    ap.add_argument("--seconds", type=float, default=60, help="--test: clip seconds")
    ap.add_argument("--max-seconds", type=int, default=300, help="live: Nuance session length (self-drive: 0 = "
                    "until Ctrl+C)")
    ap.add_argument("--out", default=None, help="record what you see + hear (side by side with the driver) to this mp4")
    ap.add_argument("--voice-host", default="joyai", help="GPU machine for the voice ('' = this laptop)")
    ap.add_argument("--mic", default="default", help="live: PulseAudio source you talk with (to the persona, or "
                    "the voice that becomes the character's with a self driver)")
    ap.add_argument("--url", default="ws://127.0.0.1:18080/ws")
    ap.add_argument("--dashboard", type=int, default=8765)
    ap.add_argument("--no-face-pass", action="store_true")
    ap.add_argument("--no-voice", action="store_true", help="plain persona voice (no conversion)")
    ap.add_argument("--bridge-args", default="", help="extra live_bridge.py flags")
    a = ap.parse_args()
    c = load_character(a.character)
    v, conv = c["voice"], c["conversation"]
    self_drive = a.driver in SELF
    pitch = a.pitch if a.pitch is not None else (v.get("pitch_self", v["pitch"]) if self_drive else v["pitch"])
    if self_drive and not a.test and a.max_seconds == 300:
        a.max_seconds = 0  # you drive: run until Ctrl+C
    tmp = REPO / ".tmp" / "character"
    tmp.mkdir(parents=True, exist_ok=True)
    out = a.out or (str(REPO / "test-outputs" / f"{a.character}-{time.strftime('%m%d-%H%M')}.mp4")
                    if a.test else None)

    live_self = self_drive and not a.test        # your own camera + mic, no driver process of ours for the audio
    mods = [] if live_self else [sink("char_audio")]
    env = dict(os.environ)
    if a.phone:      # the character's voice -> phone_out (-> the phone); the phone's mic -> phone_mic
        mods += [sink("phone_out"), sink("phone_mic")]
        env["PULSE_SINK"] = "phone_out"
        a.mic = "phone_mic.monitor"
    elif a.test:
        mods.append(sink("char_out"))
        env["PULSE_SINK"] = "char_out"
    run_s = a.seconds if a.test else a.max_seconds
    if live_self:
        inp = a.camera if a.driver == "webcam" else f"srt://0.0.0.0:{a.srt_port}"
        mon = a.mic
    else:
        inp, mon = f"srt://0.0.0.0:{PORT}", "char_audio.monitor"
    bridge = [PY, str(REPO / "cloud/client/live_bridge.py"), "--url", a.url, "--input", inp,
              "--portrait", "--ref", str(REPO / c["appearance"]["reference"]),
              "--prompt", (REPO / c["appearance"]["prompt_file"]).read_text().strip(),
              "--audio-monitor", "--monitor-source", mon,
              "--voice", "off" if a.no_voice else v["kind"], "--voice-model", str(REPO / v["model"]), "--voice-pitch", str(pitch),
              "--voice-index-rate", str(v["index_rate"]), "--voice-block-ms", str(v["block_ms"]),
              "--voice-protect", str(v.get("protect", 0.5)), "--voice-rms-mix", str(v.get("rms_mix", 1.0)),
              # test: a fixed length; live: run until the driver ends (we stop the bridge then), so waiting for the
              # phone / a slow start never cuts the conversation short
              "--seconds", str(int(run_s + 40)) if a.test else "0"]
    if a.voice_host and not a.no_voice:
        bridge += ["--voice-host", a.voice_host]
    if not a.no_face_pass:
        bridge += ["--facefix-preset", c["face_pass"]["lpfast_preset"]]
    bridge += ["--output", str(tmp / "played.mp4")] if a.test else ["--output", "ffplay"]
    if a.phone or not a.test:
        bridge += ["--dashboard", str(a.dashboard)]
    if out:
        bridge += ["--record-av", out, "--record-av-sbs"]
    bridge += shlex.split(a.bridge_args)
    log = open(tmp / "bridge.log", "w")
    procs = []
    try:
        print(f"[character] {c['name']}: driver {a.driver}{' (test clip)' if a.test else ''}, voice pitch {pitch}; "
              f"starting the bridge (log {tmp / 'bridge.log'})", flush=True)
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
        if a.phone:
            pv = subprocess.Popen([PY, str(REPO / "tools/phone_view.py"), "--out-sink", "phone_out", "--mic-sink",
                                   "phone_mic", "--port", str(a.phone_port), "--dashboard", str(a.dashboard),
                                   "--title", c["name"]] + (["--camera-port", str(CAM_PORT)] if a.driver == "nuance" else []), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            procs.insert(0, pv)
            for line in pv.stdout:
                print(line.rstrip(), flush=True)
                if "https://" in line or "Traceback" in line:
                    break
            phone_in = threading.Event()

            def relay():
                for x in pv.stdout:
                    print(x.rstrip(), flush=True)
                    if "[phone] connected" in x:
                        phone_in.set()
            threading.Thread(target=relay, daemon=True).start()
            if not a.test:
                print("[character] open that link on the phone, accept the certificate warning once, tap Start "
                      "(the session starts when the phone connects; waiting up to 15 min)", flush=True)
                if a.driver == "nuance" and not phone_in.wait(900):
                    sys.exit("[character] no phone connected in 15 min")
        if a.test:
            clip = a.clip or str(REPO / (c.get("self_drive", {}).get("test_clip", "assets/self_drive_test.mp4")
                                         if self_drive else conv["test_clip"]))
            drv = subprocess.Popen(
                ["ffmpeg", "-v", "error", "-re", "-i", clip, "-t", str(a.seconds), "-map", "0:v", "-c:v", "libx264",
                 "-preset", "veryfast", "-tune", "zerolatency", "-g", "30", "-b:v", "5M", "-f", "mpegts",
                 f"srt://127.0.0.1:{PORT}?mode=caller&latency=20", "-map", "0:a", "-ac", "1", "-f", "pulse",
                 "-device", "char_audio", "character-clip"])
        elif live_self:
            print(f"[character] you drive {c['name']}: wear HEADPHONES (your converted voice plays back in sync with "
                  f"the output window; speakers would feed it into the mic). Ctrl+C to stop.", flush=True)
            if a.driver == "phone":
                print("[character] iPhone: Control Center -> Screen Mirroring -> JoyAI-Cam, camera app full screen",
                      flush=True)
                drv = subprocess.Popen(["bash", str(REPO / "tools/iphone_airplay.sh")],
                                       env=dict(os.environ, SRT_PORT=str(a.srt_port)))
            else:
                if a.driver == "srt":
                    print(f"[character] start your phone's SRT stream to srt://<laptop IP>:{a.srt_port} (caller)",
                          flush=True)
                drv = br     # nothing else to run: wait for the bridge (Ctrl+C or --max-seconds)
        else:
            drv = subprocess.Popen(
                [NPY, str(REPO / "tools/nuance_live/nuance_source.py"), "--srt", f"srt://127.0.0.1:{PORT}",
                 "--sink", "char_audio", "--persona", conv["persona_id"],
                 "--prompt", (REPO / conv["personality_file"]).read_text().strip(), "--say", conv["first_line"],
                 "--expressiveness", str(conv.get("expressiveness", 1.0)), "--mic", a.mic,
                 "--max-seconds", str(a.max_seconds)]
                + (["--camera", f"tcp://127.0.0.1:{CAM_PORT}"] if a.phone else []))
        if drv is not br:
            procs.append(drv)
        try:   # the clip streamer can hang on closing its SRT connection after the clip: don't wait for it
            drv.wait(timeout=a.seconds + 10 if a.test else (a.max_seconds + 30 if a.max_seconds else None))
        except subprocess.TimeoutExpired:
            drv.terminate()
        time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
                try:
                    p.wait(180)   # the bridge writes the --out recording at exit (~1 s per 2 s of video)
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
