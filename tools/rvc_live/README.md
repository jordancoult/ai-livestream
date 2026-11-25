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

## Where to run it: not on the laptop GPU when lpfast runs (measured 2026-11-25)
lpfast alone uses ~92% of the RTX 5080 Laptop (37 ms per frame at 25 fps). Any voice stage on the same GPU pushes it
over: face-pass frames rendered 95% (voice off) -> 24% (100 ms blocks), 78% (300 ms); with CUDA MPS 66-90%, and
an MPS SM cap (`--voice-gpu-share`) 84-97% (noisy). CPU is far too slow (780 ms per 300 ms block).
**Run it on the JoyAI 5090 instead:** `cloud/infra/voice_up.sh joyai` once, then `--voice rvc --voice-host joyai
--voice-block-ms 200`: face pass 96.8%, JoyAI chunk time +3% (fps unchanged), glass-to-glass unchanged (~955 ms); the
stage delay incl. the SSH round trip (~0.2-0.4 s) stays far below the video's ~0.95 s, so the voice adds no latency.
