# LongLive 1 on a rented RTX 5090 (Scope + bridge, same stack as the laptop)

Same Scope fork (`jordancoult/scope`, branch `blackwell-realtime`), same `tools/scope_bridge.py`, same patches as the
laptop. The only difference is the hardware profile: `./start_scope.sh --hw 5090` (or `HW_PROFILE=5090`). Results and the
chosen presets: `experiments/2026-11-20-ll1-5090/FINDINGS.md`.

## Setup on a fresh box (~5 min + ~16 GB of downloads)
```bash
git clone https://github.com/jordancoult/ai-livestream.git /workspace/ail
git clone -b blackwell-realtime https://github.com/jordancoult/scope.git /workspace/ail/scope
cd /workspace/ail && bash cloud/pipelines/longlive1/setup_box.sh    # uv sync (torch 2.9.1 cu128, Sage, flash-attn)
HF_HUB_ENABLE_HF_TRANSFER=0 bash cloud/pipelines/longlive1/models_box.sh   # LongLive, Wan VAE/T5, VACE, TAEs, depth, CLIP
```
Then `./start_scope.sh --hw 5090` (server) and the bridge with `--hw 5090`. The profile:
- uses the 5090 preset table (native 832×480);
- skips the laptop-only parts: CPU pinning, thermal watchdog, battery/v4l2 checks.

## Measuring (all run on the box, from /workspace/ail)
| Script | What |
|---|---|
| `bench_box.sh <tag> [bench_longlive args]` | Model-only fps + frame-aligned clip (`tools/bench_longlive.py`) with the laptop's LongLive env |
| `sweep_box.sh grid.txt [args]` | Runs a list of configs through `bench_box.sh` (detach it: `setsid nohup ...`) |
| `score_box.sh out.txt "<prompt>" input.mp4 clips...` | `tools/quality_metrics.py` + prompt match |
| `live_box.sh <tag> [bridge args]` | Real Scope server + bridge, flash-file input, glass-to-glass latency (`DEPTH=1` for depth). Measure latency at noise ≤ 0.6, because at 1.0 the flashes don't reach the output |
| `live_laptop.sh` | The same live measurement on the laptop, for comparison |

Live use from the laptop webcam still needs the network path. WebRTC from the laptop bridge to Scope on the box needs
UDP, or the JoyAI-style WebSocket relay; that isn't built yet.
