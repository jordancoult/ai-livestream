# Audio-driven expressions for the lpfast face pass (Ditto as the driver) — 2026-11-23

Slot: `cloud/lpfast/audio_driver.py` (`AudioMotion`): Ditto (antgroup/ditto-talkinghead, Apache-2.0 code + weights)
predicts LivePortrait motion at 25 fps from speech (streaming HuBERT ONNX + LMDM diffusion, PyTorch); only the
expression feeds lpfast, which keeps its renderer, pose gates and paste onto the JoyAI frames. Offline:
`run_offline.py --audio speech.wav --audio-src <human face image> [--preset gstar] [--set JSON]`.

| | |
|---|---|
| Speed | 394 motion frames in 8.5 s on the 5080 (46 fps, 50 sampling steps) -> real-time headroom |
| Effect always on | faces_ok 99.9 % (no webcam gates: covering your mouth / turning away no longer switches it off) vs 90 % webcam-driven |
| Lip sync | mouth articulates per syllable (open/close/teeth, `mouth_strip` check); gstar gains (lip 2.65) = bigger, gains 1 = subtler |
| Source image | Ditto's face detector doesn't find G-Star's face (cartoon): it is conditioned on a human face (a webcam frame); lpfast's relative mode transfers the motion onto G-Star |

Sample: `compare_audio.mp4` (30 s, with sound): JoyAI | lpfast from webcam | audio-driven (gstar gains) | audio-driven
(gains 1). The speech is Ditto's example clip (a stranger's voice over your recorded body motion).

Licences: Ditto Apache-2.0; its avatar registration uses InsightFace det_10g / 2d106det (non-commercial) once per
source image -> replace for the product. Live (next): Ditto's online mode with mic chunks (0.2 s hop) feeding the bridge.
