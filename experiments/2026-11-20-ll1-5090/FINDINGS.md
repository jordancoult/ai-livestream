# LongLive 1 on a rented RTX 5090: best config (2026-11-20)

**Answer:** native 832×480, 3 steps, stock memory, fp8, `lighttae` = laptop fps and latency (21 fps, 1.68 s) with a far
sharper world. Use `./start_scope.sh --hw 5090`, preset 20; depth presets 15/24 keep you recognisable in the world.

The question: what's the best LongLive 1 setup on the cloud 5090 for the high-noise "worlds" use? That's webcam
pose/appearance at noise 0.85–1.0, prompt only, like the saves `pure-worlds` and `interactive-worlds-2`. It used the
laptop stack (Scope fork + bridge + patches) unchanged.
- Budget (owner): about the same GPU latency as the laptop; the JoyAI 0.55 s cap does not apply here.
- Input: the portrait webcam recording `bench/webcam/input.mp4`. Prompt "POV flying over a cyber city", noise 1.0.
- Recipe: fp8 row-wise, compiled blocks + TAE, lighttae, 12-frame chunks.
- Files: `results.jsonl` (model-only), `quality_scores.txt`, `live_5090.txt`, `live_laptop.txt`. Scripts:
  `cloud/pipelines/longlive1/`.

## Result: same fps and latency as the laptop, with much more image
Live through Scope + bridge (latency = flash in → flash out, including the 14-frame output cushion):

| | Size, steps | Live fps | Latency |
|---|---|---|---|
| **Laptop**, `pure-worlds` | 576×320, 2 | 20.4 | 1.66 s |
| 5090, same settings | 576×320, 2 | 30 (input-capped) | 1.09 s |
| **5090 "world detail"** (preset 20) | 768×432, 4 | 20.6 | 1.67 s |
| **5090 "you in the world"** (preset 15) | 768×432, 2 + depth | 20.8–23.6 | 1.55 s |

## Model-only speed (fps, chunk time)
| Size | 1 step | 2 steps | 3 steps | 4 steps |
|---|---|---|---|---|
| 576×320 | 62.6 / 0.19 s | 45.5 / 0.26 s | 35.0 / 0.34 s | 28.2 / 0.43 s |
| 768×432 | 46.5 / 0.26 s | 34.0 / 0.35 s | 26.9 / 0.45 s | 22.1 / 0.54 s |
| 832×480 (native) | 36.6 / 0.33 s | 26.7 / 0.45 s | 20.8 / 0.58 s | 17.2 / 0.70 s |
| 960×544 | 26.5 / 0.45 s | 19.6 / 0.61 s | | |

**Depth costs about 30%:** 576/2 30.4, 768/1 34.2, 768/2 24.2, 832/1 27.6, 832/2 19.6 fps. VRAM: 15 GB at 576×320,
22 GB at 832×480, 26 GB at 960×544.

## Quality (`quality_metrics.py`, all scored at 320×576 against the same input)
- **Depth is the biggest lever.**
  - Adherence 0.88 vs 0.3–0.5, about 3× less flicker and invented motion, and the best realism. The top 3 configs
    overall are all depth 2-step.
  - By eye: you stay recognisable (face, glasses → goggles, pose) in front of the world.
  - The prompt match is lower (≈20–23 vs ≈24–25), because the world is a backdrop rather than everything.
- **Without depth** (the `pure-worlds` look: you turn into the scene, e.g. a drone-pilot rig), more steps and size win.
  - The best are 768×432 at 4 steps (score 1.20) and 960×544 at 2 steps (1.18). The laptop config, 576×320 at 2
    steps, scores 0.55.
  - By eye, 768/4 has a coherent city (streets, cars, towers) where 576/2 is mush.
- **1 step is clearly worse** at every size without depth (realism 0.13–0.27).
- **Rejected:**
  - 8- and 4-frame chunks (`SCOPE_FRAMES_PER_BLOCK=2/1`): lower latency, but worse realism/prompt match/drift, and 4
    frames halves the fps;
  - bf16 instead of fp8: no better, 32% slower;
  - fused KV: bit-identical output, no speed gain on the 5090.

## Follow-up: native resolution, longer memory, better decoder (same day)
- **Native 832×480 is the sharpest by eye.** Compared as full-resolution crops at the 720×1280 output size:
  - 3 steps gives distinct towers, street grids and lights;
  - 2 steps adds a little ghosting;
  - 768×432 has a streaky "glitch" texture.
  `quality_metrics.py` scores at 320×576, so it can't see this. Its slight preference for 768 isn't a quality signal.
- **Longer memory is worse** (`SCOPE_LOCAL_ATTN_SIZE`, stock 12 latent frames ≈ 1.6 s; see
  `quality_scores_native.txt`):
  - Scores fall from 12 to 18 to 24 at 2 and 3 steps, with more invented motion and lower realism.
  - By eye, window 24 goes blurry, streaky and darker after about 15 s while 12 stays crisp. The model was trained at
    12.
  - It also costs speed: 18 frames −16%, 24 frames −28%; +1.7 GB VRAM per 6 frames.
- **Full decoder:** the Wan VAE and `lightvae` look the same as `lighttae` at native size. The Wan VAE halves the fps
  (9.9), `lightvae` costs 11%. Keep `lighttae`.
- **So the spare VRAM has no quality use** for LongLive 1 here; the 5090's headroom is best spent on resolution and
  steps.
- **Depth at native:** 3 steps isn't better than 2 (score 2.40 vs 2.48) and drops to 15 fps.

Live at native 832×480 (laptop `pure-worlds` reference: 20.4 fps, 1.66 s):

| Steps | Live fps | Latency |
|---|---|---|
| **3** | 20.8–21.0 | 1.68 s |
| 2 | 27.2 | 1.31 s |
| 2 + depth | 19.8 | 1.78 s |

## Chosen 5090 presets (`scope_bridge.py --hw 5090`, native 832×480)
| Slider | Steps | Depth | Live fps |
|---|---|---|---|
| 30 | 2 | off | 27.2 |
| 24 | 1, fast "you in the world" | on | ~25 (model 27.6) |
| **20** | **3, best world** (laptop fps and latency) | off | **21.0** |
| 15 | 2, "you in the world" | on | 19.8 |

The laptop table and behaviour are unchanged, checked by launching `start_scope.sh` without the flag:
- same CPU pinning, threads and PRIME env;
- the watchdog and virtual mic start as before;
- no profile variables are set.

## Notes
- The first chunk returns 9 frames instead of 12, so clips are offset by 3 frames (0.1 s) against the input.
  That's negligible for these metrics.
- The first sweep's clips were blown out to white by a `bench_longlive.py` writer bug (LongLive now returns uint8).
  The fps numbers were valid. Fixed, and the scores above come from the re-run.
- Not done yet: live use from the laptop webcam to the box (network path), and drift beyond the 17 s scored clips.
