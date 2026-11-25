# rvc_live: realtime voice conversion as an audio stage (local GPU)

RVC v2 voice models (e.g. `models/rvc/karen/`, gitignored) run through Applio's realtime core (MIT, pinned in
`setup.sh`, cloned to `cloud/third_party/applio`). Used by the bridge as `--audio-monitor --voice rvc`
(`cloud/client/voice_stage.py`): the voice is converted **inside the A/V delay**, so it adds no latency while the
video pipeline is slower than the stage (JoyAI ~0.9 s vs the stage ~0.17 s).

| | RTX 5080 Laptop |
|---|---|
| compute per 100 ms block | 15 ms mean, 17 ms p95 |
| algorithmic delay (measured at startup, speech-like signal) | 62 ms |
| monitor delay change, voice on vs off (`monitor_delay_test.py`) | +0 ms (5 ms resolution) |
| A/V sync self-test through JoyAI, voice on vs off | audio -30 ms vs -25 ms |

- `setup.sh`: venv (torch cu128) + Applio + rmvpe/contentvec weights.
- `serve.py`: the voice-stage protocol (s16le 48 kHz in/out, `READY delay_ms=…`); any other converter can implement it
  and plug in with `--voice-cmd`.
- `offline_stream_test.py IN OUT`: block-by-block conversion of a file, timing + delay.
- `monitor_delay_test.py CLIP`: virtual mic -> monitor -> virtual output, delay with the stage off vs on.
- Pitch: `--voice-pitch` (+5 male -> Karen per the model's README, 0 for a female speaker).

Non-commercial experiment (the RVC models are trained on a real person's voice; use with their consent only).
