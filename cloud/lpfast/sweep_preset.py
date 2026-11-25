"""Sweep face-pass settings offline on a recorded JoyAI run and score each against the driver (face_metrics.py).

  cloud/facefix/.venv/bin/python cloud/lpfast/sweep_preset.py --run cloud/lpfast/experiments/2026-11-25-karen \
      --base girl --max 500 --cfg 'eye1:{"gains":{"eye":1.0}}' --cfg 'lip13:{"gains":{"lip":1.3}}'

Scores (driver = the input face, result = the face pass output):
  gy_sd / gy_mean: vertical gaze spread / mean (eyes rolling up/down = no eye contact; driver's for reference)
  mouth_corr: how well the mouth follows the driver; mouth_scale: result opening / driver opening (p95, 1 = same)
  eye_corr: blinks follow the driver
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from face_metrics import compare, measure  # noqa: E402

PY = sys.executable


def cols(review, dst):
    out = []
    for k in range(3):
        p = dst / f"col{k}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(review), "-vf", f"crop=iw/3:ih:{k}*iw/3:0", "-c:v",
                        "libx264", "-crf", "14", str(p)], check=True)
        out.append(str(p))
    return out


def gaze(rows):
    r = [x for x in rows if x]
    gy = np.array([x["gy"] for x in r])
    gx = np.array([x["gx"] for x in r])
    return {"gy_mean": float(gy.mean()), "gy_sd": float(gy.std()), "gx_mean": float(gx.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with joyai_raw.mp4, input.mp4, input.json (bridge --save-input)")
    ap.add_argument("--base", default="girl")
    ap.add_argument("--max", type=int, default=500)
    ap.add_argument("--cfg", action="append", default=[], help="name:JSON overrides (run_offline --set)")
    a = ap.parse_args()
    run = Path(a.run)
    drv = None
    for c in a.cfg:
        name, js = c.split(":", 1)
        dst = run / "sweep" / name
        r = subprocess.run([PY, str(HERE / "run_offline.py"), "--out", str(run / "joyai_raw.mp4"), "--inp",
                            str(run / "input.mp4"), "--stats", str(run / "input.json"), "--dst", str(dst), "--mode",
                            "abs", "--preset", a.base, "--set", js, "--max", str(a.max)], capture_output=True, text=True)
        if r.returncode:
            print(name, "FAILED", r.stderr[-500:])
            continue
        c0, _, c2 = cols(dst / "review.mp4", dst)
        if drv is None:
            drv = measure(c0)
            print("driver", json.dumps({k: round(v, 3) for k, v in gaze(drv).items()}))
        res = measure(c2)
        m = compare(drv, res)
        m.update(gaze(res))
        keep = ("gy_mean", "gy_sd", "gx_mean", "mouth_corr", "mouth_scale", "width_scale", "eye_corr")
        print(f"{name:14s}", json.dumps({k: round(m[k], 3) for k in keep}), flush=True)
        (dst / "metrics.json").write_text(json.dumps(m))


if __name__ == "__main__":
    main()
