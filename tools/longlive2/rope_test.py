"""Long-run check: does the stream survive past latent frame 1024 (chunk 128), and how does it look late in the run?
  LL2_ROOT=/workspace /root/ll2/bin/python tools/longlive2/rope_test.py --rope relative --chunks 160 --out /workspace/rt_rel"""
import argparse, os, sys, time
import cv2
sys.path.insert(0, os.path.dirname(__file__))
p = argparse.ArgumentParser()
p.add_argument("--rope", default="absolute"); p.add_argument("--no-kvq", action="store_true")
p.add_argument("--refresh", type=int, default=0); p.add_argument("--refresh-strength", type=float, default=0.5)
p.add_argument("--chunks", type=int, default=160)
p.add_argument("--height", type=int, default=480); p.add_argument("--width", type=int, default=832)
p.add_argument("--out", required=True); p.add_argument("--video", action="store_true")
p.add_argument("--prompt", default="a cinematic tracking shot following a woman walking through a busy night market, neon signs")
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
from ll2_stream import LL2Engine
eng = LL2Engine(height=a.height, width=a.width, ckpt="S2", kv_quant=not a.no_kvq, rope=a.rope,
                refresh_every=a.refresh, refresh_strength=a.refresh_strength)
eng.reset(prompt=a.prompt)
vw = cv2.VideoWriter(f"{a.out}/run.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 24, (a.width, a.height)) if a.video else None
t0 = time.time()
for c in range(a.chunks):
    try:
        fr = eng.step()
    except Exception as e:
        print(f"CRASH at chunk {c} (latent frame {eng.start}): {type(e).__name__}: {str(e)[:200]}", flush=True); sys.exit(1)
    if vw:
        for f in fr: vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    if c % 10 == 0 or c == a.chunks - 1:
        cv2.imwrite(f"{a.out}/c{c:04d}.jpg", cv2.cvtColor(fr[16], cv2.COLOR_RGB2BGR))
        print(f"chunk {c} ok, {eng.stats['total_ms']:.0f} ms, {time.time()-t0:.0f}s", flush=True)
if vw: vw.release()
print("DONE", flush=True)
