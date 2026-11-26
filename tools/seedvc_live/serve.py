"""Seed-VC voice-stage server: zero-shot voice conversion from ONE short clip of the target voice (1-30 s), for
characters without a trained RVC model. Same protocol as tools/rvc_live/serve.py (cloud/client/voice_stage.py):

  stdin : s16le mono 48 kHz          stdout: s16le mono 48 kHz, sample-for-sample, written per block
  stderr: logs; one 'READY delay_ms=<algorithmic delay> block=<samples>' line once warm

Runs inside a clone of https://github.com/Plachtaa/seed-vc (GPL-3.0; the box: /workspace/seed-vc, its .venv; setup:
tools/seedvc_live/setup_box.sh). Uses the repo's real-time model (DiT uvit-tat + XLSR, "tiny") and ports the
streaming loop of its real-time-gui.py (context windows, SOLA crossfade) at 48 kHz, so its blocks are our stream's.
An RMS gate replaces the GUI's FunASR VAD (silence in -> silence out, no hallucinated breaths).

  cd /workspace/seed-vc && .venv/bin/python serve.py --ref /workspace/seedvc_refs/<name>.wav [--steps 6]
"""
import argparse
import importlib.util
import os
import sys
import time
import types

import numpy as np

SR = 48000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="target voice clip (any format librosa reads), 1-30 s")
    ap.add_argument("--steps", type=int, default=6, help="diffusion steps (4-10: quality vs time per block)")
    ap.add_argument("--block-ms", type=int, default=180)
    ap.add_argument("--crossfade-ms", type=int, default=40)
    ap.add_argument("--extra-ce", type=float, default=2.5, help="content-encoder left context (s)")
    ap.add_argument("--extra-dit", type=float, default=0.5, help="DiT left context (s)")
    ap.add_argument("--extra-right", type=float, default=0.02, help="right context (s): adds this much delay")
    ap.add_argument("--cfg", type=float, default=0.7, help="inference_cfg_rate")
    ap.add_argument("--max-prompt", type=float, default=3.0, help="seconds of the clip used as the prompt")
    ap.add_argument("--gate-db", type=float, default=-50.0, help="blocks quieter than this -> silence")
    ap.add_argument("--gain", type=float, default=0.0, help="output gain in dB")
    ap.add_argument("--pitch", type=float, default=0, help="(accepted for compatibility; the tiny model has no F0 "
                    "control)")
    ap.add_argument("--index-rate", "--protect", "--rms-mix", dest="_ignored", default=None)
    a = ap.parse_args()
    out_fd = os.dup(1)
    os.dup2(2, 1)                      # the repo prints a lot: stdout carries audio only

    import librosa
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat

    spec = importlib.util.spec_from_file_location("rtgui", os.path.join(os.getcwd(), "real-time-gui.py"))
    rt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rt)        # module level only (its GUI lives under __main__)
    dev = torch.device("cuda")
    rt.device = dev
    t0 = time.time()
    model_set = rt.load_models(types.SimpleNamespace(checkpoint_path=None, config_path=None, fp16=True, gpu=0))
    msr = model_set[-1]["sampling_rate"]
    ref = librosa.load(a.ref, sr=msr)[0]
    print(f"[seedvc] models loaded in {time.time() - t0:.1f} s (model sr {msr}); reference {len(ref) / msr:.1f} s",
          flush=True)

    # --- streaming state, as real-time-gui.py start_vc() with samplerate = 48 kHz ---
    zc = SR // 50
    fr = lambda s: int(np.round(s * SR / zc)) * zc  # noqa: E731
    block = fr(a.block_ms / 1000)
    block16 = 320 * block // zc
    xfade = fr(a.crossfade_ms / 1000)
    sola_buf_n, sola_search = min(xfade, 4 * zc), zc
    extra, extra_r = fr(a.extra_ce), fr(a.extra_right)
    input_wav = torch.zeros(extra + xfade + sola_search + block + extra_r, device=dev)
    input_res = torch.zeros(320 * input_wav.shape[0] // zc, device=dev)
    sola_buffer = torch.zeros(sola_buf_n, device=dev)
    skip_head, skip_tail = extra // zc, extra_r // zc
    ret_len = (block + sola_buf_n + sola_search) // zc
    fade_in = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=sola_buf_n, device=dev)) ** 2
    fade_out = 1 - fade_in
    to48 = tat.Resample(orig_freq=msr, new_freq=SR).to(dev) if msr != SR else None
    gain = 10 ** (a.gain / 20)

    def step(x):
        nonlocal sola_buffer
        input_wav[:-block] = input_wav[block:].clone()
        input_wav[-block:] = torch.from_numpy(x).to(dev)
        input_res[:-block16] = input_res[block16:].clone()
        input_res[-320 * (block // zc + 1):] = torch.from_numpy(librosa.resample(
            input_wav[-block - 2 * zc:].cpu().numpy(), orig_sr=SR, target_sr=16000)[320:]).to(dev)
        with torch.no_grad():
            y = rt.custom_infer(model_set, ref, a.ref, input_res, block16, skip_head, skip_tail, ret_len, a.steps,
                                a.cfg, a.max_prompt, a.extra_ce - a.extra_dit)
            if to48 is not None:
                y = to48(y)
        if 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9) < a.gate_db:
            y = torch.zeros_like(y)
        # SOLA (as the GUI): align the new block to the previous tail, crossfade
        conv_in = y[None, None, :sola_buf_n + sola_search]
        nom = F.conv1d(conv_in, sola_buffer[None, None, :])
        den = torch.sqrt(F.conv1d(conv_in ** 2, torch.ones(1, 1, sola_buf_n, device=dev)) + 1e-8)
        off = int(torch.argmax(nom[0, 0] / den[0, 0]).item())
        y = y[off:]
        y[:sola_buf_n] = y[:sola_buf_n] * fade_in + sola_buffer * fade_out
        sola_buffer = y[block:block + sola_buf_n].clone()
        return y[:block].float().cpu().numpy()

    for _ in range(3):                 # warm up (first runs pay CUDA init / cache setup)
        step(np.zeros(block, np.float32))
    tw = time.time()
    for _ in range(5):
        step((0.05 * np.random.default_rng(0).standard_normal(block)).astype(np.float32))
    per_block = (time.time() - tw) / 5 * 1000
    mem = torch.cuda.max_memory_allocated() / 2 ** 30
    # algorithmic delay beyond the block itself: right context + SOLA crossfade/search window
    delay_ms = (extra_r + sola_buf_n + sola_search) / SR * 1000
    print(f"READY delay_ms={delay_ms:.0f} block={block} warmup_ms={per_block:.1f} per_block_ms={per_block:.1f} "
          f"peak_vram_gb={mem:.2f}", file=sys.stderr, flush=True)
    inp, out = sys.stdin.buffer, os.fdopen(out_fd, "wb", buffering=0)
    while True:
        b = inp.read(block * 2)
        if len(b) < block * 2:
            return
        y = step(np.frombuffer(b, np.int16).astype(np.float32) / 32768) * gain
        out.write((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())


if __name__ == "__main__":
    main()
