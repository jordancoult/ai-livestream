"""Voice-stage server for cloud/client/voice_stage.py (protocol, any converter can implement it):

  stdin : s16le mono 48 kHz, any write size
  stdout: s16le mono 48 kHz, the converted stream, sample-for-sample (same count as read, written per block)
  stderr: logs; one line 'READY delay_ms=<algorithmic delay> block=<samples>' once the model is warm

  tools/rvc_live/.venv/bin/python tools/rvc_live/serve.py --model models/rvc/karen/karen.pth --pitch 5
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--index", default=None)
    ap.add_argument("--pitch", type=int, default=5)
    ap.add_argument("--index-rate", type=float, default=0.5)
    ap.add_argument("--block-ms", type=int, default=100)
    ap.add_argument("--protect", type=float, default=0.5, help="protect voiceless consonants (0..0.5, 0.5 = off)")
    ap.add_argument("--rms-mix", type=float, default=1.0, help="volume envelope: 1 = the model's loudness, lower = "
                    "follow the input's")
    ap.add_argument("--gain", type=float, default=0.0, help="output gain in dB")
    a = ap.parse_args()
    out_fd = os.dup(1)
    os.dup2(2, 1)  # library prints go to stderr; stdout carries audio only
    from rvc_engine import RVCEngine, DEFAULT_MODEL
    eng = RVCEngine(model_path=a.model or DEFAULT_MODEL, index_path=a.index, block_ms=a.block_ms, pitch=a.pitch,
                    index_rate=a.index_rate, protect=a.protect, rms_mix=a.rms_mix)
    gain = 10 ** (a.gain / 20)
    print(f"READY delay_ms={eng.delay_ms} block={eng.block} warmup_ms={eng.warmup_ms:.1f}", file=sys.stderr, flush=True)
    inp, out = sys.stdin.buffer, os.fdopen(out_fd, "wb", buffering=0)
    nbytes = eng.block * 2
    while True:
        b = inp.read(nbytes)
        if len(b) < nbytes:
            return
        x = np.frombuffer(b, np.int16).astype(np.float32) / 32768
        y = eng.process(x) * gain
        out.write((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main()
