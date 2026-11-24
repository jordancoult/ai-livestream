"""Audio-driven expression driver for the lpfast face pass (experimental slot, 2026-11-23).

Ditto (antgroup/ditto-talkinghead, Apache-2.0; ACM MM 2025) predicts LivePortrait motion (expression, head angles) at
25 fps from speech. Its renderer isn't used: lpfast keeps rendering + pasting onto the JoyAI frames, and only its
expression *source* changes from the webcam to the audio (the user's mic or an AI voice).

  from audio_driver import AudioMotion
  am = AudioMotion()                                 # loads HuBERT (streaming ONNX) + LMDM (PyTorch)
  frames = am.from_wav("speech.wav", "assets/char_face.png")   # list of per-frame dicts at 25 fps
  drv = am.as_driver(frames[i])                      # lp_pass driver dict (gates pass; exp = audio's)

Models: cloud/lpfast/checkpoints/ditto (HF digital-avatar/ditto-talkinghead: ditto_pytorch/{models,aux_models},
ditto_cfg). Note: Ditto's avatar registration uses InsightFace det/landmark models (non-commercial licence); they only
run once on the character image.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DITTO = HERE / "third_party" / "ditto"
CKPT = HERE / "checkpoints" / "ditto"
sys.path.insert(0, str(DITTO))


def _preload_cuda13():
    """ONNX Runtime's CUDA provider here is built for CUDA 13; torch (and lpfast) use CUDA 12. Load the CUDA 13 cuBLAS/
    runtime from a side folder (cloud/facefix/cu13libs, `uv pip install --target`) so Ditto's HuBERT runs on the GPU
    instead of silently falling back to the CPU (0.9 s -> faster audio-to-motion)."""
    import ctypes
    lib = HERE.parent / "facefix" / "cu13libs" / "nvidia" / "cu13" / "lib"
    cur = Path(sys.prefix) / "lib" / "python3.12" / "site-packages" / "nvidia" / "curand" / "lib"
    for f in (lib / "libcudart.so.13", lib / "libcublasLt.so.13", lib / "libcublas.so.13", cur / "libcurand.so.10"):
        try:
            ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass


_preload_cuda13()


class _Null:
    """Stands in for Ditto's renderer stages (warp / decode / putback / writer / stitch): not needed here."""
    def __init__(self, *a, **k):
        self.d0 = None

    def __call__(self, *a, **k):
        return None

    def setup(self, *a, **k):
        pass

    def set_Nd(self, *a, **k):
        pass

    def close(self):
        pass


class AudioMotion:
    def __init__(self, cfg="v0.4_hubert_cfg_pytorch.pkl", sampling_timesteps=None, online=False):
        import importlib
        spo = importlib.import_module("stream_pipeline_online" if online else "stream_pipeline_offline")
        for name in ("MotionStitch", "WarpF3D", "DecodeF3D", "PutBack", "VideoWriterByImageIO"):
            setattr(spo, name, _Null)
        cwd = os.getcwd()
        os.chdir(DITTO)
        try:
            self.sdk = spo.StreamSDK(str(CKPT / "ditto_cfg" / cfg), str(CKPT / "ditto_pytorch"))
        finally:
            os.chdir(cwd)
        self.sampling_timesteps = sampling_timesteps
        self.frames = []

        def stitch_worker(sdk=self.sdk, out=self.frames):  # collect the motion instead of rendering it
            import queue
            while not sdk.stop_event.is_set():
                try:
                    item = sdk.motion_stitch_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    sdk.warp_f3d_queue.put(None)
                    break
                fid, x_d_info, _ = item
                out.append((fid, {k: np.asarray(v, np.float32) for k, v in x_d_info.items()}))
                if hasattr(out, "on_frame"):
                    out.on_frame(fid)
        self.sdk._motion_stitch_worker = stitch_worker

    def from_wav(self, wav, source_img, **setup_kwargs):
        """Speech file -> list (one per 25 fps frame) of {'exp': (1, 63), 'pitch'/'yaw'/'roll', 'scale', 't'}."""
        import librosa
        if self.sampling_timesteps:
            setup_kwargs.setdefault("sampling_timesteps", self.sampling_timesteps)
        self.frames.clear()
        tmp = str(HERE / ".audio_driver_null.mp4")
        self.sdk.setup(str(source_img), tmp, **setup_kwargs)
        audio, _ = librosa.load(str(wav), sr=16000)
        n = math.ceil(len(audio) / 16000 * 25)
        self.sdk.setup_Nd(N_d=n)
        self.sdk.audio2motion_queue.put(self.sdk.wav2feat.wav2feat(audio))
        self.sdk.close()
        for p in (tmp, tmp + ".tmp.mp4"):
            if os.path.exists(p):
                os.remove(p)
        return [d for _, d in sorted(self.frames, key=lambda x: x[0])]

    @staticmethod
    def as_driver(f, device="cuda"):
        """A per-frame motion dict -> the lp_pass driver dict. The webcam gates see a frontal, visible face (always
        open); the character's own pose gates still apply."""
        return {"exp": torch.from_numpy(f["exp"].reshape(1, 21, 3)).to(device).float(), "R": None, "iod": 1.0,
                "yaw": 0.0, "pitch": 0.0, "mouth_vis": 1.0, "face_score": 1.0, "audio": True}


class _Frames(list):
    """List of (frame index, motion) that also reports arrivals (LiveAudioMotion)."""
    on_frame = None


class LiveAudioMotion:
    """Live audio -> LivePortrait motion with Ditto's streaming pipeline: 0.2 s audio hops, 10-frame (0.4 s) motion
    windows, 10 sampling steps. Audio comes from a PulseAudio/PipeWire source through ffmpeg (no extra deps) or a file
    played in real time (tests). motion_at(t) returns the motion for wall-clock time t (blocking up to a timeout).

      live = LiveAudioMotion(face_img, source="default")   # or source="file:speech.wav"
      live.start(); drv = AudioMotion.as_driver(live.motion_at(time.time() - 0.3))
    """
    CS = (3, 5, 2)  # (context, hop, lookahead) in 40 ms units

    def __init__(self, face_img, source="default", steps=5, offset_frames=0, window=10):
        import threading
        self.am = AudioMotion(online=True)
        self.am.frames = _Frames()
        self.frames = self.am.frames
        # re-bind the collector to the new list (the closure captured the old one)
        sdk = self.am.sdk
        out = self.frames

        def stitch_worker(sdk=sdk, out=out):
            import queue
            while not sdk.stop_event.is_set():
                try:
                    item = sdk.motion_stitch_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    sdk.warp_f3d_queue.put(None)
                    break
                _, x_d_info, _ = item  # (its first field is the SOURCE frame index: 0 for a still image)
                out.append((self.n_out, {k: np.asarray(v, np.float32) for k, v in x_d_info.items()}))
                self.n_out += 1
                with self.cond:
                    self.cond.notify_all()
        sdk._motion_stitch_worker = stitch_worker
        self.cond = threading.Condition()
        self.n_out = 0
        self.source, self.face_img, self.steps, self.window = source, face_img and str(face_img), steps, window
        self.offset = offset_frames      # measured alignment between audio time and motion frame index
        self.t0 = None                   # wall-clock time of audio sample 0
        self.proc = None
        self.stats = {"chunks": 0, "late": 0}

    def start(self):
        import threading
        tmp = str(HERE / ".audio_driver_live_null.mp4")
        # window = new motion frames per generation (80-frame sequence, overlap = 80 - window): 10 = 0.4 s
        self.am.sdk.setup(self.face_img, tmp, online_mode=True, overlap_v2=80 - self.window,
                          sampling_timesteps=self.steps)
        self.am.sdk.setup_Nd(N_d=10 ** 7)
        threading.Thread(target=self._feed, daemon=True).start()
        return self

    def _reader(self):
        """Yields float32 16 kHz mono blocks of 640 samples (40 ms) and sets t0."""
        import subprocess
        import time
        if self.source.startswith("file:"):
            import librosa
            audio, _ = librosa.load(self.source[5:], sr=16000)
            self.t0 = time.time()
            for i in range(0, len(audio) - 640, 640):
                while time.time() < self.t0 + i / 16000:
                    time.sleep(0.002)
                yield audio[i:i + 640]
            return
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "pulse", "-fragment_size", "1280", "-i", self.source,
               "-ac", "1", "-ar", "16000", "-f", "f32le", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
        import atexit
        atexit.register(lambda: self.proc.poll() is None and self.proc.kill())
        n = 0
        while True:
            b = self.proc.stdout.read(640 * 4)
            if len(b) < 640 * 4:
                return
            if self.t0 is None:
                self.t0 = time.time() - 0.04
            n += 1
            yield np.frombuffer(b, np.float32)

    def _feed(self):
        c, h, la = self.CS
        split = int(sum(self.CS) * 0.04 * 16000) + 80
        buf = np.zeros(c * 640, np.float32)  # the context Ditto expects before sample 0
        pos = 0
        for blk in self._reader():
            buf = np.concatenate([buf, blk])
            while len(buf) - pos >= split:
                self.am.sdk.run_chunk(buf[pos:pos + split], self.CS)
                self.stats["chunks"] += 1
                pos += h * 640
            if pos > 640 * 200:  # keep the buffer small
                buf, pos = buf[pos - 0:], 0

    def close(self):
        """Stop Ditto's worker threads (they are not daemons: without this the process never exits)."""
        self.am.sdk.stop_event.set()
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()

    def motion_at(self, t, timeout=1.0):
        """Motion for wall-clock time t (None until the stream has started)."""
        import time
        if self.t0 is None or not self.frames:
            return None
        idx = int(round((t - self.t0) * 25)) + self.offset
        deadline = time.time() + timeout
        with self.cond:
            while self.frames[-1][0] < idx and time.time() < deadline:
                self.cond.wait(timeout=0.05)
        if self.frames[-1][0] < idx:
            self.stats["late"] += 1
            return self.frames[-1][1]
        # frames arrive in order; index from the end
        j = len(self.frames) - 1 - (self.frames[-1][0] - idx)
        return self.frames[max(0, min(len(self.frames) - 1, j))][1]



def _serve(face_img, source, steps, window):
    """`python audio_driver.py serve FACE SOURCE STEPS WINDOW`: Ditto streaming in its own process. Binary records on
    the original stdout: b'T' + float64 t0, then b'F' + int32 index + 63 float32 per motion frame. Logs go to stderr."""
    import struct
    import time
    out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)  # Ditto's prints -> stderr
    live = LiveAudioMotion(face_img, source=source, steps=int(steps), window=int(window))
    base = live.frames.append

    def append(x):
        base(x)
        out.write(b"F" + struct.pack("<i", x[0]) + np.asarray(x[1]["exp"], np.float32).reshape(-1).tobytes())
        if len(live.frames) > 200:
            del live.frames[:100]
    live.frames.append = append
    live.start()
    while live.t0 is None:
        time.sleep(0.02)
    out.write(b"T" + struct.pack("<d", live.t0))
    while True:
        time.sleep(1)


class LiveAudioMotionProc:
    """LiveAudioMotion in its own process: Ditto's Python threads then don't compete with the bridge's camera / network
    threads for the GIL (in-process it cut the camera loop to ~17-20 fps). Same API: face_img, start(), motion_at(t),
    close(), stats."""

    def __init__(self, face_img=None, source="default", steps=5, window=5):
        import threading
        self.face_img, self.source, self.steps, self.window = face_img, source, steps, window
        self.frames = []          # (index, {"exp": array}) like LiveAudioMotion
        self.t0 = None
        self.cond = threading.Condition()
        self.stats = {"late": 0, "frames": 0}
        self.p = None

    def start(self):
        import subprocess
        import threading
        self.p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "serve", str(self.face_img),
                                   self.source, str(self.steps), str(self.window)],
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        import atexit
        atexit.register(self.close)
        threading.Thread(target=self._recv, daemon=True).start()
        return self

    def _recv(self):
        import struct
        rd = self.p.stdout
        while True:
            tag = rd.read(1)
            if not tag:
                return
            if tag == b"T":
                self.t0 = struct.unpack("<d", rd.read(8))[0]
                continue
            idx = struct.unpack("<i", rd.read(4))[0]
            exp = np.frombuffer(rd.read(63 * 4), np.float32).copy()
            with self.cond:
                self.frames.append((idx, {"exp": exp}))
                self.stats["frames"] += 1
                if len(self.frames) > 3000:
                    del self.frames[:1000]
                self.cond.notify_all()

    def close(self):
        if self.p is not None and self.p.poll() is None:
            self.p.kill()

    def motion_at(self, t, timeout=1.0):
        import time
        if self.t0 is None or not self.frames:
            return None
        idx = int(round((t - self.t0) * 25))
        deadline = time.time() + timeout
        with self.cond:
            while self.frames[-1][0] < idx and time.time() < deadline:
                self.cond.wait(timeout=0.05)
            if self.frames[-1][0] < idx:
                self.stats["late"] += 1
                return self.frames[-1][1]
            j = len(self.frames) - 1 - (self.frames[-1][0] - idx)
            return self.frames[max(0, min(len(self.frames) - 1, j))][1]


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "serve":
    _serve(*sys.argv[2:6])
