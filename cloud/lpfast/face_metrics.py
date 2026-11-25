"""Face metrics for tuning the face pass against its driver (MediaPipe FaceMesh with iris landmarks).

Per frame: gaze (iris centre within the eye, x: -1 left .. +1 right, y: -1 up .. +1 down, 0 = looking straight),
mouth opening (inner-lip gap / inter-ocular distance), mouth width, eye opening. Compares a result video with the
driver frame by frame (same frame count / pairing).

  cloud/facefix/.venv/bin/python cloud/lpfast/face_metrics.py --driver input_paired.mp4 --video lp.mp4 [--video joyai.mp4]
"""
import argparse
import json

import cv2
import mediapipe as mp
import numpy as np

L_EYE = (33, 133, 159, 145, 468)    # outer corner, inner corner, upper lid, lower lid, iris centre (subject's right)
R_EYE = (362, 263, 386, 374, 473)
LIP_IN = (13, 14)
LIP_CORNERS = (61, 291)


def measure(path, max_frames=0):
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                         min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap, rows = cv2.VideoCapture(path), []
    while True:
        ok, f = cap.read()
        if not ok or (max_frames and len(rows) >= max_frames):
            break
        r = fm.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if not r.multi_face_landmarks:
            rows.append(None)
            continue
        p = np.array([(l.x * f.shape[1], l.y * f.shape[0]) for l in r.multi_face_landmarks[0].landmark])
        gx, gy, eo = [], [], []
        for o, i, u, d, c in (L_EYE, R_EYE):
            a, b = p[o], p[i]
            ax = b - a
            w = np.linalg.norm(ax)
            t = np.dot(p[c] - a, ax) / w ** 2           # 0..1 along the eye
            n = np.array([-ax[1], ax[0]]) / w
            mid = (p[u] + p[d]) / 2
            h = max(1e-3, np.linalg.norm(p[u] - p[d]))
            gx.append((t - 0.5) * 2 * (1 if o == 33 else -1))   # + = towards the subject's left in both eyes
            gy.append(np.dot(p[c] - mid, n) / h)
            eo.append(h / w)
        iod = np.linalg.norm(p[468] - p[473])
        rows.append({"gx": float(np.mean(gx)), "gy": float(np.mean(gy)), "eye": float(np.mean(eo)),
                     "mouth": float(np.linalg.norm(p[LIP_IN[0]] - p[LIP_IN[1]]) / iod),
                     "width": float(np.linalg.norm(p[LIP_CORNERS[0]] - p[LIP_CORNERS[1]]) / iod)})
    return rows


def compare(drv, res):
    pairs = [(d, r) for d, r in zip(drv, res) if d and r]
    if len(pairs) < 20:
        return {"frames": len(pairs)}
    D = {k: np.array([d[k] for d, _ in pairs]) for k in pairs[0][0]}
    R = {k: np.array([r[k] for _, r in pairs]) for k in pairs[0][1]}
    c = lambda a, b: float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else 0.0  # noqa: E731
    return {
        "frames": len(pairs),
        "gaze_err": float(np.mean(np.hypot(R["gx"] - D["gx"], R["gy"] - D["gy"]))),   # vs the driver's gaze
        "gaze_off_center": float(np.mean(np.abs(R["gx"]))),
        "drv_gaze_off_center": float(np.mean(np.abs(D["gx"]))),
        "mouth_corr": c(D["mouth"], R["mouth"]),
        "mouth_scale": float(np.percentile(R["mouth"], 95) / max(1e-3, np.percentile(D["mouth"], 95))),
        "width_scale": float(np.median(R["width"]) / np.median(D["width"])),
        "width_p95": float(np.percentile(R["width"], 95)),
        "eye_corr": c(D["eye"], R["eye"]),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", required=True)
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    drv = measure(a.driver, a.max)
    for v in a.video:
        print(v, json.dumps({k: round(x, 3) if isinstance(x, float) else x for k, x in compare(drv, measure(v, a.max)).items()}))
