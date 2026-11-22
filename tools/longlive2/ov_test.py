"""Overlap-trick check: same prompt/input with --overlap 0 vs 4 (offline, native size). Writes an mp4 per run + timing.
  LL2_ROOT=/workspace /root/ll2/bin/python tools/longlive2/ov_test.py --overlap 4 --seconds 20 --out /workspace/ov"""
import argparse, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
p = argparse.ArgumentParser()
p.add_argument("--overlap", type=int, default=4); p.add_argument("--seconds", type=float, default=20)
p.add_argument("--height", type=int, default=704); p.add_argument("--width", type=int, default=1280)
p.add_argument("--out", required=True); p.add_argument("--input", default=None, help="video file: v2v")
p.add_argument("--strength", type=float, default=0.9)
p.add_argument("--prompt", default="a cinematic shot of a woman with red hair talking and laughing in a sunlit cafe")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
from ll2_stream import LL2Engine
eng = LL2Engine(height=a.height, width=a.width, ckpt="S2", kv_quant=True, overlap=a.overlap)
src = None
if a.input:
    cap = cv2.VideoCapture(a.input); src = []
    while True:
        ok, f = cap.read()
        if not ok: break
        h, w = f.shape[:2]; s = max(a.width / w, a.height / h)
        f = cv2.resize(f, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        y0, x0 = (f.shape[0] - a.height) // 2, (f.shape[1] - a.width) // 2
        src.append(cv2.cvtColor(f[y0:y0 + a.height, x0:x0 + a.width], cv2.COLOR_BGR2RGB))
tag = f"{'v2v' if src else 't2v'}_ov{a.overlap}"
W = a.width * (2 if src else 1)
vw = cv2.VideoWriter(f"{a.out}/{tag}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (W, a.height))
for warm in range(2):  # compile warm-up, then the measured run from a fresh stream
    eng.reset(prompt=a.prompt); pos, made, times = 0, 0, []
    while made < (24 * a.seconds if warm else 96):
        need = eng.frames_needed()
        kw = {}
        if src:
            kw = {"input_frames": [src[(pos + i) % len(src)] for i in range(need)], "strength": a.strength}
        t = time.perf_counter(); fr = eng.step(**kw); times.append(time.perf_counter() - t)
        if warm:
            for i, f in enumerate(fr):
                o = np.hstack([kw["input_frames"][min(i, need - 1)], f]) if src else f
                vw.write(cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
        pos += len(fr); made += len(fr)
vw.release()
st = sorted(times[2:])
med = st[len(st) // 2]
print(f"RESULT {tag}: {len(times)} steps, {len(fr)} frames/step, median {1000*med:.0f} ms/step -> "
      f"{len(fr)/med:.1f} fps gen, chunk {len(fr)/24:.2f} s", flush=True)
