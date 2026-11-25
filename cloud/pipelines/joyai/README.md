# Pipeline: JoyAI-Video-Edit

[JoyAI-Video-Edit](https://github.com/jd-opensource/JoyAI-Video-Edit) (JD, Apache-2.0): a 16B MM-DiT streaming
video editor with a MiMo-VL-7B condition encoder and a causal VAE. It edits live frames in 8-frame chunks
from an instruction plus an optional reference image (RV2V). We use the upstream server
(`deploy/xvideo/serving/serve_joyomni_streaming.py`, FastAPI WebSocket on :8080) unchanged and drive it with our
own clients.

## Requirements
- **GPU: Blackwell sm_120 with ≥ 32 GB**: RTX 5090 (target), RTX PRO 6000, RTX PRO 4500 (works at ~half speed).
  `setup.sh` checks it. The settings are picked per GPU in `run_server.sh` (5090: SageAttention + fp16-accumulate
  FP8 + low-VRAM layout; PRO 6000: cuDNN attention, full-rate accumulate, normal layout).
- Host RAM ≥ 48 GB (the low-VRAM layout keeps the text encoder on the CPU). Disk ≥ 110 GB free under `$JOYAI_DATA`.
- NVIDIA driver for CUDA ≥ 12.8; `setup.sh` installs the CUDA 12.8 compiler with apt (Ubuntu 22.04/24.04).

## Files
| File | What |
|---|---|
| `setup.sh` | `artifacts` (private HF wheels + compile cache) · `env` (CUDA compiler, venv, kernels) · `weights` (public HF) · `all` |
| `run_server.sh` | Starts the server: per-GPU settings, face++ checkpoint, compile cache restored from `$JOYAI_DATA/cache/<GPU>.tar` |
| `save_cache.sh` | After the first warm start on a new GPU model: packs the compile cache into one tar |
| `publish_artifacts.sh` | Uploads wheels + cache tars to `JordanCo/ai-livestream-artifacts` (needs a write token) |
| fork [`jordancoult/JoyAI-Video-Edit`](https://github.com/jordancoult/JoyAI-Video-Edit/tree/blackwell-realtime) | Branch `blackwell-realtime` = upstream + our commits: no YOLO body gate outside full-body detection; web-UI **memory panel** (**Reset (R)** key/button, since-reset timer, auto-reset interval, default 0 = never); **runtime LoRA** (off unless `JOYAI_LORA` is set); speed work (below); a dev kernel profiler (`JOYOMNI_TORCH_PROFILE`). `setup.sh` clones it; rebase on `upstream/main` to take JD's updates |
| `prompts/` | Working prompts (G-Star, human) |
| `bench/` | Box-local throughput benchmark (`bench_box.sh`), block parity test (`parity.py`, original vs patched DiT block, bit-exact), FP8 GEMM tile sweep (`gemm_bench.py`) |
| `experiments/` | Dated runs with FINDINGS.md (videos are gitignored) |

## Speed on the RTX 5090 (2026-11-19, `bench/`)
Measured box-local with `bench/bench_box.sh`, input fed faster than real time. The old `joyai_client.py` capped
input at the clip's 24 fps, so earlier "26 fps" figures were input-bound.

| | 40 fps input | 30 fps input (webcam-like) | server latency |
|---|---|---|---|
| upstream | 25.8 fps | 26.6 fps | 0.53–0.60 s |
| fork (default) | 27.3–27.7 fps | 27.3 fps | 0.51–0.55 s |
| ~~fork + `"backpressure": "hold"`~~ (rejected: over the 0.55 s budget) | 28.2 fps | 28.3–28.8 fps | 0.68–0.72 s |

**Fork speed commits** (all lossless: `bench/parity.py` shows a real-size DiT block is bit-identical to upstream):
- one K/V concat per layer and step instead of two;
- FP8 GEMM tiles re-tuned for the 5090;
- adaLN norm kernel reads the per-sample scale/shift directly instead of expanded [tokens, D] copies (22 → 8 µs per
  call; JIT-built once per GPU, about 40 s, into the compile cache);
- GELU fused into the FP8 quant of each MLP down-projection (exhaustively checked over all bf16 values).

**Where the time goes:** the GPU is saturated; the stages already overlap on separate CUDA streams. Per 8-frame
chunk:
- FP8 GEMMs ~140 ms. JD's Triton kernel runs at 450–600 TFLOPS, faster than cuBLAS and near the card's limit;
- VAE ~65–80 ms;
- attention ~31 ms;
- the rest ~40 ms.

**Frame admission:**
- With the default 2 chunks in flight, camera frames that arrive while the pipeline is full are dropped. The DiT then
  idles about 40 ms per chunk waiting for 8 fresh frames.
- **Latency budget: 0.55 s (owner's rule). Keep the defaults.** Two options trade latency for fps and are rejected
  for going over it:
  - `"backpressure": "hold"` keeps the newest 7 frames so the next chunk starts at once: +1.5–2 fps but 0.68–0.72 s.
    It stays in the fork, off by default;
  - `max_inflight_chunks` 3: 28.0 fps, 0.76 s.
- Any fps gained by queueing work ahead of the DiT costs exactly that queue's wait in latency, so a future speedup
  has to come from less GPU work per chunk.

**Not worth it:** CUDA graph off (same fps). **Lossless headroom left:** small, a few % (VAE convs run cuDNN sm80
kernels; q/k copies). Bigger gains need lossy changes (NVFP4, fewer tokens or steps) or a faster GPU.

## G-Star detail LoRA (optional, off by default)
Adapters from `training/` (better G-Star skin/hat detail; mouth unchanged; `cloud/docs/gstar-lora-finetune.md`).
Applied at runtime next to the FP8/bf16 linears (merging would lose it):
```bash
/root/venv/bin/hf download JordanCo/joyai-gstar-lora adapters/r3_step50.pt --local-dir /workspace/lora   # private
JOYAI_LORA=/workspace/lora/adapters/r3_step50.pt JOYAI_LORA_SCALE=1.0 bash run_server.sh
```
The log shows `#####[runtime_lora] ... 320 FP8 + 0 bf16 linears hooked`. Speed cost on the 5090 (same box,
2026-11-04): base 0.319 s per 8-frame chunk (25 fps) · full LoRA 0.38 s (~21 fps) · `JOYAI_LORA_MODULES=attn_qkv,attn_proj`
0.352 s (22.7 fps). The hooks add separate small bf16 GEMMs and re-read each layer's input; a fused kernel would be needed to
make it free. Run the bridge with `--fps`/`--play-fps` near the server rate. About 65% of the LoRA's
change survives the bf16 add (vs ~30% merged into bf16, ~7% merged into FP8).

## Recipe (what works)
- **Reference:** a full-body character image on a plain background (`assets/char_front.png`). An upper-body
  crop turned G-Star into a realistic human torso.
- **Prompt (rule since 2026-11-25, the user's call): JoyAI's own character-replacement preset for every character
  swap: "Replace me with the character in the reference picture."** (`prompts/replace.txt`; `gstar.txt`, `karen.txt`,
  `human_blonde.txt` hold the same). With the 1.1 weights it works well; the earlier finding (explicit descriptions
  beat the generic prompt, `prompts/gstar_hatfix.txt`) was on the 1.0 weights. The built-in prompt enhancer (`use_pe`) made identity worse, so it's off.
- **Checkpoint `1101_face++`** (released 2026-11-01) over `0811`: clearly better human identity and motion.
- **Session options** (start message): `detection_mode: "off"`, `use_pe: false`, `kv_reset_frames: 0` (no drift
  in 2 min; the default 600 resets every 25 s without a visible jump), H.264 both ways.

## Numbers (RTX 5090, 840×480 or 480×840, 2 steps)
| | |
|---|---|
| Output fps | 25–27 (30 fps camera in); per chunk: DiT 215 ms, VAE enc 26 ms, dec 46 ms → ~26.5 fps ceiling |
| Latency on the GPU machine | 0.46 s median, 0.58 s p95 |
| Glass-to-glass, laptop → EU → laptop, smooth playout | 1.03 s median, 1.07 s p95 (2-min run, 0 drops) |
| RTX PRO 4500 (half a 5090) | ~10–11 fps, 1.15 s on the machine |

Details: `experiments/2026-11-02-joyai-first/FINDINGS.md` (prompt, ref and checkpoint tests on the PRO 4500),
`experiments/2026-11-02-joyai-5090/FINDINGS.md` (5090 speed, codecs, portrait, live bridge, stability).

## Known limits / next
- Hands stay human-shaped (pink) on G-Star; the head comes out rounder than the reference's tall cone.
- 840×480 is the 5090's real-time limit (720p needs a B200: sm_100, not supported by our kernels yet).
- First start on a new GPU model compiles ~12 min; publish its cache tar afterwards.
