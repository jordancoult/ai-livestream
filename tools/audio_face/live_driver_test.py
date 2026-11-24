"""Latency + alignment of LiveAudioMotion on a speech file played in real time (vs Ditto offline on the same file)."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "cloud", "lpfast"))
from audio_driver import AudioMotion, LiveAudioMotion
face, wav = sys.argv[1], sys.argv[2]
arr = {}
live = LiveAudioMotion(face, source="file:" + wav, steps=int(os.environ.get("STEPS", "5")), window=int(os.environ.get("WIN", "10")))
base_append = live.frames.append
def append(x):
    base_append(x); arr[x[0]] = time.time()
live.frames.append = append
live.start()
dur = __import__("librosa").get_duration(path=wav)
time.sleep(dur + 2)
idxs = sorted(arr)
lat = np.array([arr[i] - (live.t0 + i / 25) for i in idxs])
print(f"frames {len(live.frames)} chunks {live.stats['chunks']}; latency (motion frame ready - its audio time): "
      f"median {np.median(lat):.2f} s, p90 {np.percentile(lat, 90):.2f}, max {lat.max():.2f}, first index {idxs[0]}", flush=True)
off = AudioMotion().from_wav(wav, face)
d = np.load(os.path.join(os.path.dirname(__file__), "..", "..", "cloud", "lpfast", "lip_open_dir.npy")).ravel()
ix = [k * 3 + c for k in (6, 12, 14, 17, 19, 20) for c in (0, 1, 2)]
lv = np.array([f["exp"].ravel()[ix] @ d for _, f in sorted(live.frames, key=lambda x: x[0])])
ov = np.array([f["exp"].ravel()[ix] @ d for f in off])
res = []
for s in range(-20, 21):
    a = lv[max(0, s):]; b = ov[max(0, -s):]; n = min(len(a), len(b), 300)
    res.append((np.corrcoef(a[:n], b[:n])[0, 1], s))
best = max(res)
print(f"alignment vs offline: best shift {best[1]} frames (corr {best[0]:.2f}); corr at 0: {dict((s, c) for c, s in res)[0]:.2f}", flush=True)
os._exit(0)
