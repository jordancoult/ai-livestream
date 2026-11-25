# JoyAI-VL-Interaction as the character's "brain" (mode D) — 2026-11-23

Model: jdopensource/JoyAI-VL-Interaction (8B, Apache-2.0): watches a video stream at 1 frame/s and decides every second
to stay silent / speak / delegate. Their stack (vLLM 0.22 + adapter; Qwen3-TTS via vLLM-Omni) on one rented RTX 5090.
Test: the user's webcam clip (60 s) replayed in real time (`stream_test.py`), three conditions, then the replies voiced
with Qwen3-TTS (voice "ryan") and put on G-Star through Ditto -> lpfast (audio-driven face, `cloud/lpfast/audio_driver.py`).
Videos: `test-outputs/joyai-vl-persona.mp4`, `joyai-vl-questions.mp4` (webcam | G-Star co-host, captions, voice).

## Behaviour (INT4)
| Run | Result |
|---|---|
| Proactive, no prompt (`--no-force-silence-before-query`) | silent all 60 s (faces/gestures not "noteworthy" for the default prompt); 0.19 s per decision |
| 4 spoken-style questions | all 4 correct (tank top/hair/beard; pointing at chest; touching chin and neck looking up; surprised), 0.17-0.27 s |
| G-Star co-host persona (standing prompt) | spoke at 0, 20, 35, 50, 55 s = the expressive moments (one odd line at 0 s), playful lines, 0.13-0.44 s |
| Qwen3-TTS | first audio 0.06-0.12 s, ~6x faster than real time -> moment -> voice ~0.3-0.5 s |

## Quantizations on an RTX 5090 (sm_120)
| Variant | Weights | Runs? | Notes |
|---|---|---|---|
| NVFP4 (official) | 7.4 GB | **no** | "No supported CUDA architectures found for major versions [12]": their vLLM build lacks sm_120 NVFP4 kernels |
| INT4 (official, W4A16 g32) | 7.4 GB | **yes** | needs `VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer: "requires sm75 or higher" on sm_120) |
| FP8 / INT8 (official) | 10.2 GB | yes | same backend flags; on this box only 12k context fit -> requests failed after ~40 s; persona answered in **Chinese** (INT4 English) |
| BF16 | ~17 GB | not tested | needs a clean 32 GB GPU |

Box caveat: the rented 5090 had ~15.8 GB held by no visible process (host-side leak), so everything ran in ~15.5 GB.
On a clean 5090 the summary model (Qwen3-VL-4B, long-term memory every 100 s) and FP8/INT8 with full context fit.
Setup notes: their audio runtime install needed `UV_HTTP_TIMEOUT=600`; the TTS venv needs its own `nvidia/cu13/lib` on
LD_LIBRARY_PATH; TTS stage memory lowered to 0.22 each.

## Fit for us
- Proactive, scene-aware co-host that talks only when something happens — the strength PersonaPlex lacks (blind).
- Speech goes through text (TTS), so no natural overlap; latency is still low (~0.3-0.5 s to voice).
- Laptop 5080 (16 GB): INT4 (7.4 GB) should fit, but it shares the GPU with lpfast/Ditto. A small cloud GPU is cleaner.
