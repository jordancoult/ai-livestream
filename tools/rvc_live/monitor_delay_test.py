"""Check that the voice stage adds no delay to the A/V monitor: a speech clip plays into a virtual mic, the monitor
delays it by a fixed target, and the actual input->output delay is measured with the voice stage off and on.
Uses virtual PulseAudio sinks only (no mic, no speakers).

  cloud/facefix/.venv/bin/python tools/rvc_live/monitor_delay_test.py assets/clip_with_speech.mp4 [--target-ms 800]
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "cloud", "client"))
from av_sync import AudioMonitor, PLAY_LATENCY_MS  # noqa: E402
from voice_stage import VoiceStage, build_cmd  # noqa: E402

SR = 48000


def env(x, hop=240):
    n = len(x) // hop
    return np.log(np.sqrt((x[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)) + 1e-3)


def lag_ms(a, b, max_ms=2000, hop=240):
    ea, eb = env(a, hop), env(b, hop)
    ea, eb = ea - ea.mean(), eb - eb.mean()
    best = (-9, 0)
    for k in range(0, int(max_ms / (hop / SR * 1000))):
        n = min(len(ea), len(eb) - k)
        if n < 200:
            break
        c = float(np.corrcoef(ea[:n], eb[k:k + n])[0, 1])
        best = max(best, (c, k))
    return best[1] * hop / SR * 1000, best[0]


def run(clip, target, voice, seconds):
    rec = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "pulse", "-fragment_size", "960", "-i", "vt_in.monitor",
                            "-f", "pulse", "-fragment_size", "960", "-i", "vt_out.monitor", "-filter_complex",
                            "[0:a]pan=mono|c0=c0[a0];[1:a]pan=mono|c0=c0[a1];[a0][a1]amerge=inputs=2[a]", "-map", "[a]", "-ac", "2", "-ar", str(SR), "-t",
                            str(seconds + 3), "-f", "s16le", "-"], stdout=subprocess.PIPE)
    os.environ["PULSE_SINK"] = "vt_out"
    mon = AudioMonitor("vt_in.monitor", lambda: target, voice).start()
    time.sleep(0.5)
    play = subprocess.Popen(["ffmpeg", "-v", "error", "-re", "-i", clip, "-t", str(seconds), "-vn", "-ac", "1",
                             "-f", "pulse", "-device", "vt_in", "clip"])
    raw = rec.stdout.read()
    play.wait(); mon.close(); rec.wait()
    st = np.frombuffer(raw, np.int16).reshape(-1, 2)
    d, c = lag_ms(st[:, 0], st[:, 1])
    return d, c, dict(mon.stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--target-ms", type=float, default=800)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--pitch", type=int, default=0)
    a = ap.parse_args()
    mods = [subprocess.check_output(["pactl", "load-module", "module-null-sink", f"sink_name={n}"]).decode().strip()
            for n in ("vt_in", "vt_out")]
    try:
        expect = a.target_ms + PLAY_LATENCY_MS
        d0, c0, s0 = run(a.clip, a.target_ms, None, a.seconds)
        print(f"voice OFF: measured {d0:.0f} ms (target {a.target_ms:.0f} + playback ~{PLAY_LATENCY_MS}), "
              f"corr {c0:.2f}, {s0}", flush=True)
        v = VoiceStage(build_cmd("rvc", pitch=a.pitch))
        d1, c1, s1 = run(a.clip, a.target_ms, v, a.seconds)
        print(f"voice ON : measured {d1:.0f} ms, corr {c1:.2f}, {s1}, stage {v.stats()}")
        print(f"voice stage changes the delay by {d1 - d0:+.0f} ms (expected ~0; envelope resolution 5 ms)")
        _ = expect
    finally:
        for m in mods:
            subprocess.run(["pactl", "unload-module", m])


if __name__ == "__main__":
    main()
