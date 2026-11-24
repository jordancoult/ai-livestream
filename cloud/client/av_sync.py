"""Audio/video sync helpers for live_bridge.py.

AudioMonitor: your mic, delayed by the live total latency, played to the default output (headphones!) so the local
preview (ffplay) and your voice line up. Capture/playback through ffmpeg (pulse) + pacat, no extra Python deps. The
delay follows a callable (ms) smoothly: 10 ms blocks are dropped or silence is added when it drifts by > 30 ms.

Camera latency probe: a full-screen window flashes black/white (`python av_sync.py blink ...`, its own process); point
the camera (webcam or phone) at it; the bridge records the input brightness at each frame's arrival and the lag that
best matches the flash pattern = camera -> bridge latency (sensor, AirPlay/USB, encode, network, decode). Saved to
.tmp/camera_latency.json and added to the bridge's own glass-to-glass time for the total.
"""

import collections
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CAL_FILE = REPO / ".tmp" / "camera_latency.json"
SR, BLOCK = 48000, 480  # 10 ms blocks


class AudioMonitor:
    def __init__(self, source, get_delay_ms):
        self.source, self.get_delay_ms = source, get_delay_ms
        self.stop = threading.Event()
        self.stats = {"delay_ms": None, "dropped": 0, "inserted": 0}

    def start(self):
        self.cap = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-f", "pulse", "-fragment_size", "1920", "-i",
                                     self.source, "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                                    stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        self.play = subprocess.Popen(["pacat", "--raw", "--format=s16le", f"--rate={SR}", "--channels=1",
                                      "--latency-msec=30", "--stream-name=bridge-monitor"], stdin=subprocess.PIPE)
        threading.Thread(target=self._run, daemon=True).start()
        return self

    def _run(self):
        buf = collections.deque()
        silence = bytes(BLOCK * 2)
        while not self.stop.is_set():
            b = self.cap.stdout.read(BLOCK * 2)
            if len(b) < BLOCK * 2:
                return
            buf.append(b)
            target = self.get_delay_ms()
            if target is None:
                continue  # until the first latency measurement: keep buffering, play nothing
            target_blocks = max(1, int(target / 10))
            self.stats["delay_ms"] = round(target)
            if len(buf) < target_blocks - 1:      # too little delay: play 10 ms of silence, let the buffer grow
                out = [silence]
                self.stats["inserted"] += 1
            else:
                while len(buf) > target_blocks + 1:   # too much: drop the oldest 10 ms
                    buf.popleft()
                    self.stats["dropped"] += 1
                out = [buf.popleft()]
            try:
                for o in out:
                    self.play.stdin.write(o)
            except (BrokenPipeError, ValueError):
                return

    def close(self):
        self.stop.set()
        for p in (getattr(self, "cap", None), getattr(self, "play", None)):
            if p is not None and p.poll() is None:
                p.kill()


def load_camera_latency():
    try:
        return json.loads(CAL_FILE.read_text())
    except (OSError, ValueError):
        return None


def save_camera_latency(ms, source, score):
    CAL_FILE.parent.mkdir(exist_ok=True)
    d = {"ms": round(ms), "source": source, "score": round(score, 3), "measured": time.strftime("%Y-%m-%d %H:%M:%S")}
    CAL_FILE.write_text(json.dumps(d))
    return d


def start_blink(seconds, log_path):
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "blink", str(seconds), str(log_path)])


def analyze(samples, log_path, max_lag_ms=1500):
    """samples: [(arrival time, mean brightness)]; the blink log: lines 'time level'. -> (lag ms, correlation)."""
    flips = [tuple(map(float, ln.split())) for ln in Path(log_path).read_text().split("\n") if ln.strip()]
    if len(flips) < 4 or len(samples) < 30:
        return None, 0.0
    t = np.array([s[0] for s in samples]); b = np.array([s[1] for s in samples], np.float64)
    b = (b - b.mean()) / (b.std() + 1e-6)
    ft = np.array([f[0] for f in flips]); fl = np.array([f[1] for f in flips])

    def level(ts):
        i = np.searchsorted(ft, ts, side="right") - 1
        out = np.where(i >= 0, fl[np.clip(i, 0, len(fl) - 1)], np.nan)
        return out
    best = (None, -1.0)
    for lag in range(0, max_lag_ms + 1, 5):
        s = level(t - lag / 1000)
        ok = ~np.isnan(s)
        if ok.sum() < 20 or np.std(s[ok]) == 0:
            continue
        c = float(np.corrcoef(b[ok], s[ok])[0, 1])
        if c > best[1]:
            best = (lag, c)
    return best


def _blink(seconds, log_path):
    import cv2
    w = "Point the camera at this window (latency test)"
    cv2.namedWindow(w, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(w, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black = np.zeros((720, 1280, 3), np.uint8); white = np.full((720, 1280, 3), 255, np.uint8)
    rng = np.random.default_rng(1)
    t_end = time.time() + float(seconds)
    level = 0
    with open(log_path, "w") as f:
        while time.time() < t_end:
            level = 1 - level
            cv2.imshow(w, white if level else black)
            cv2.waitKey(1)
            f.write(f"{time.time():.4f} {level}\n"); f.flush()
            hold = 0.25 + 0.5 * rng.random()  # irregular intervals: one unambiguous best lag
            t_next = time.time() + hold
            while time.time() < t_next:
                cv2.waitKey(5)
    cv2.destroyAllWindows()


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "blink":
    _blink(sys.argv[2], sys.argv[3])
