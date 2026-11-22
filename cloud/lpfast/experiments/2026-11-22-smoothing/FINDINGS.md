# JoyAI + lpfast: jitter / stutter / inconsistency (2026-11-22)

Source: the user's screencast of the dashboard (webcam | JoyAI | final, ~12 fps previews). Webcam panel cropped,
upsampled to 480x840 @ 30 fps (mpdecimate + motion interpolation), replayed through JoyAI on the Virginia 5090 with the
bridge's session options (`joyai_client.py --extra '{"kv_reset_frames":0,"freeze_kv_on_static":false}'`), then lpfast
offline with the gstar preset (`run_offline.py --preset gstar [--set JSON]`, exact source_seq pairing).
Sample: `compare.mp4` (webcam aligned | JoyAI | lpfast gstar | gstar + mouth smoothing), 32 s.

| Finding | Evidence |
|---|---|
| **Stutter = camera frame rate.** Live: JoyAI received ~17 fps -> 31% of played frames repeated (1231/3989) | bridge send rate = fresh camera frames; replay at 26 fps input -> JoyAI 26 fps, 0 repeats. Webcam control `exposure_dynamic_framerate=1` (default 0) lets it drop fps in low light |
| JoyAI server itself is fine | 25.98 fps out, latency 0.55 s median (bridge options) |
| lpfast timing is fine at 26 fps input | live bridge on the clip: hit 1369/1369, render 37 ms; `--facefix-hold 4 --facefix-wait 25` did not help (91.9% hits, noise) |
| On/off of the effect in the replay = gates doing their job | hand at the chin (mouth visibility 0.40 -> cover gate), head turn (pose gate); face-score gate never closed |
| **Mouth jitter**: gstar has no smoothing on the mouth (`euro.lower = null`) at lip gain 2.65 | mouth-region temporal acceleration: JoyAI 3.31, gstar 5.04, + euro lower (0.5, 3) **4.18 (-17%)**, + stronger (lower 0.3/2, upper 1/5) 4.09 (little more) |

Caveat: the driver is the upscaled ~12 fps preview of the webcam, noisier than the real camera; live jitter is likely
lower in absolute terms, the relative gain should hold.
