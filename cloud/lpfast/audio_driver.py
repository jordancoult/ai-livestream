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
    def __init__(self, cfg="v0.4_hubert_cfg_pytorch.pkl", sampling_timesteps=None):
        import stream_pipeline_offline as spo
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
