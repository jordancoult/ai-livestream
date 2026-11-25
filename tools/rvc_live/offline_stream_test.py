"""Feed a speech file through the realtime RVC engine block by block (as the live path does) and report
per-block compute time and the end-to-end algorithmic delay (input -> output, by envelope cross-correlation).

  tools/rvc_live/.venv/bin/python tools/rvc_live/offline_stream_test.py IN.(wav|mp3|mp4) OUT.wav [--block-ms 100]
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rvc_engine import RVCEngine, SR  # noqa: E402


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).copy()


def envelope(x, hop=480):
    n = len(x) // hop
    return np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1) + 1e-12)


def delay_ms(a, b, max_ms=1000, hop=480):
    ea, eb = np.log(envelope(a, hop)), np.log(envelope(b, hop))
    ea, eb = ea - ea.mean(), eb - eb.mean()
    best, lag_best = -1e9, 0
    for lag in range(0, int(max_ms / (hop / SR * 1000))):
        n = min(len(ea), len(eb) - lag)
        c = float(np.dot(ea[:n], eb[lag:lag + n]) / n)
        if c > best:
            best, lag_best = c, lag
    return lag_best * hop / SR * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--block-ms", type=int, default=100)
    ap.add_argument("--pitch", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--extra", type=float, default=0.5)
    ap.add_argument("--protect", type=float, default=0.5)
    ap.add_argument("--rms-mix", type=float, default=1.0)
    a = ap.parse_args()
    x = load(a.inp)[: int(a.seconds * SR)]
    eng = RVCEngine(block_ms=a.block_ms, pitch=a.pitch, device=a.device, extra_s=a.extra, protect=a.protect,
                    rms_mix=a.rms_mix)
    print(f"engine ready: block {eng.block} samples, warmup {eng.warmup_ms:.1f} ms, "
          f"declared delay {eng.delay_ms:.0f} ms", flush=True)
    out, times = [], []
    for i in range(0, len(x) - eng.block + 1, eng.block):
        t = time.perf_counter()
        out.append(eng.process(x[i:i + eng.block]))
        times.append((time.perf_counter() - t) * 1000)
    y = np.concatenate(out)
    import soundfile as sf
    sf.write(a.out, y, SR)
    t = np.array(times[5:])
    print(f"blocks {len(times)}: compute mean {t.mean():.1f} ms p95 {np.percentile(t, 95):.1f} ms "
          f"max {t.max():.1f} ms (budget {a.block_ms} ms)")
    print(f"measured input->output delay (signal only, excluding compute): {delay_ms(x, y):.0f} ms")


if __name__ == "__main__":
    main()
