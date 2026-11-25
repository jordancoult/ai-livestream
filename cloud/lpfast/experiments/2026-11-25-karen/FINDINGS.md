# 2026-11-25: lpfast preset for Karen (Nuance persona mira as driver)

Input: `assets/nuance_mira_100s.mp4` (recorded Nuance session) -> JoyAI with `characters/karen/reference.png` and the
preset prompt, recorded raw (`joyai_raw.mp4`, `input.mp4`, `input.json` = source_seq; videos not in git). Offline
replay: `run_offline.py`; scoring: `face_metrics.py` (MediaPipe iris + lips) via `sweep_preset.py`.

| config | vertical gaze spread (driver 0.14) | mouth size vs driver | mouth corr | blink corr |
|---|---|---|---|---|
| JoyAI raw (no lpfast) | 0.03 | 0.38x | 0.23 | 0.24 |
| girl (lip 2.0, eye 1.3, brow 1.3, brow2 1, rest 1) | 0.26 | 1.74x | 0.95 | 0.82 |
| eye 1.0 | 0.05 | 1.74x | 0.95 | 0.85 |
| eye 1.0, lip 1.0 | 0.05 | 1.09x | 0.97 | 0.85 |
| **karen** = eye 1, lip 1, brow 1, brow2 0, rest 1 | 0.06 | 1.08x | 0.97 | 0.85 |

- The "no eye contact" look: JoyAI's Karen looks roughly straight; lpfast's eye gain 1.3 (abs mode = past the
  driver) made the irises roll (spread 0.26). Gain 1.0 fixes it.
- girl's lip gain 2 opens Karen's mouth 1.7x the driver's (grimace, bared teeth).
- brow2 / rest barely matter here.
- Cost: render 36 ms per frame on the RTX 5080 Laptop (profiled: src 11, warp+stitch 23, paste 2) = ~90% of the
  25 fps budget, which is why anything else on that GPU (e.g. RVC) makes it miss frames.
Per-config numbers: `sweep/*/metrics.json`, `karen_full/timing.json`.
