"""Realtime RVC voice conversion engine (Applio's realtime core, MIT), mono float32 at 48 kHz, fixed-size blocks.

Not model-specific to the video side: it is an audio stage. `process(block)` returns one output block of the same
size. Total input->output delay of the stage = `delay_ms` (algorithmic, measured once at startup with a synthetic
signal) + the time each block waits to fill and to compute (handled by the caller, see voice_stage.py).
"""
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
APPLIO = os.path.join(REPO, "cloud", "third_party", "applio")
SR = 48000
DEFAULT_MODEL = os.path.join(REPO, "models", "rvc", "karen", "karen.pth")


def _find_index(model_path):
    d = os.path.dirname(model_path)
    idx = [f for f in os.listdir(d) if f.endswith(".index")]
    return os.path.join(d, idx[0]) if idx else ""


class RVCEngine:
    def __init__(self, model_path=DEFAULT_MODEL, index_path=None, block_ms=100, pitch=5, index_rate=0.5,
                 protect=0.5, f0_method="rmvpe", crossfade_s=0.05, extra_s=0.5, silent_threshold_db=-60):
        if not os.path.isdir(APPLIO):
            raise SystemExit(f"Applio not found at {APPLIO}: run tools/rvc_live/setup.sh")
        cwd = os.getcwd()
        os.chdir(APPLIO)  # Applio resolves its model assets relative to the working directory
        sys.path.insert(0, APPLIO)
        try:
            from rvc.realtime.core import VoiceChanger
            self.block = int(SR * block_ms / 1000) // 480 * 480
            self.kw = dict(f0_up_key=pitch, index_rate=index_rate, protect=protect)
            self.vc = VoiceChanger(
                block_frame=self.block, cross_fade_overlap_size=crossfade_s, extra_convert_size=extra_s,
                model_path=os.path.abspath(os.path.join(cwd, model_path)),
                index_path=index_path or _find_index(model_path), f0_method=f0_method,
                embedder_model="contentvec", silent_threshold=silent_threshold_db, sid=0)
            self.warmup_ms = self.vc.warmup(**self.kw)
        finally:
            os.chdir(cwd)
        self.delay_ms = self._measure_delay()

    def process(self, block):
        out, _vol, _perf = self.vc.on_request(np.ascontiguousarray(block, np.float32), **self.kw)
        return out.astype(np.float32)

    def _reset(self):
        m = self.vc.vc_model
        for b in (m.audio_buffer, m.convert_buffer, m.pitch_buffer, m.pitchf_buffer, self.vc.sola_buffer):
            b.zero_()

    def _measure_delay(self):
        """Algorithmic delay, measured once at startup the way A/V sync is perceived: syllable-like voiced audio
        (sawtooth with a pitch contour, random syllables and pauses) goes through the stage; the lag of the output's
        log envelope vs the input's (5 ms grid, parabolic refinement) is the delay. Depends only on the block sizes."""
        rng = np.random.default_rng(3)
        x = []
        while len(x) == 0 or sum(map(len, x)) < 4 * SR:
            n = int(rng.uniform(0.08, 0.3) * SR)
            f0 = rng.uniform(140, 240) * (1 + 0.15 * np.sin(np.linspace(0, np.pi, n)))
            ph = np.cumsum(f0 / SR)
            x.append((0.25 * (2 * (ph % 1) - 1) * np.hanning(n)).astype(np.float32))
            x.append(np.zeros(int(rng.uniform(0.05, 0.25) * SR), np.float32))
        x = np.concatenate([np.zeros(self.block * 8, np.float32)] + x)
        x = x[: len(x) // self.block * self.block]
        y = np.concatenate([self.process(x[i:i + self.block]) for i in range(0, len(x), self.block)])
        self._reset()
        hop = 240

        def env(v):
            n = len(v) // hop
            e = np.log(np.sqrt((v[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1)) + 1e-3)
            return e - e.mean()
        ex, ey = env(x), env(y)
        lags = np.arange(0, int(0.4 * SR / hop))
        c = np.array([np.dot(ex[:len(ex) - k], ey[k:]) / (len(ex) - k) for k in lags])
        k = int(np.argmax(c))
        if 0 < k < len(c) - 1:
            d = c[k - 1] - 2 * c[k] + c[k + 1]
            k = k + (0.5 * (c[k - 1] - c[k + 1]) / d if d != 0 else 0)
        return round(float(k) * hop / SR * 1000)


if __name__ == "__main__":
    t = time.time()
    e = RVCEngine()
    print(f"loaded in {time.time() - t:.1f} s, warmup {e.warmup_ms:.1f} ms, delay {e.delay_ms} ms")
