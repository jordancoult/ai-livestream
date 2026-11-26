# Characters (live character mode)

A **character** is everything that makes a live persona one of ours. The live character mode
(`tools/character_live.py <name>`) turns a conversational AI persona (Nuance Live) into the character in real time:
JoyAI gives it the character's appearance, the lpfast face pass restores its expressions with the character's
tuning, RVC gives it the character's voice (in sync with the video, no added latency), and the persona talks with
the character's personality and name.

    tools/character_live.py karen            # live (headphones), output window + dashboard :8765
    tools/character_live.py karen --test     # recorded persona clip, no Nuance credits, records a review mp4

**Two drivers, same character pipeline.** The source video + source voice come either from the Nuance persona
(above) or from YOU: your camera is the source video, your mic the source voice (converted live to the character's
voice, `voice.pitch_self` semitones up, delayed to stay in sync with the output video):

    tools/character_live.py karen --driver webcam            # laptop webcam (--camera /dev/videoN) + mic
    tools/character_live.py karen --driver phone             # iPhone camera via AirPlay mirroring (starts
                                                             # tools/iphone_airplay.sh; mirror to "JoyAI-Cam") + mic
    tools/character_live.py karen --driver srt               # phone app streaming SRT to port 9000 (Larix) + mic
    tools/character_live.py karen --driver webcam --test     # self-drive test clip (assets/self_drive_test.mp4)
    options: --mic SOURCE (e.g. a headset mic; pactl list short sources), --pitch N, --bridge-args '--input-rotate cw'

**From your phone** (`--phone`, either driver): the laptop serves a page on the Wi-Fi (`tools/phone_view.py`, HTTPS
with a self-signed certificate: accept the warning once; the link carries a token). It shows the character's video
and plays its voice, in sync; the phone's mic is your voice (to the Nuance persona, or the source voice with a self
driver), and with the Nuance driver the phone's front camera goes to the persona too (`camera: true`: it sees you). With the Nuance driver the session starts only when the phone connects (no credits spent before).

    tools/character_live.py karen --phone

Wear headphones (your converted voice plays back in sync; speakers would feed it into the mic). Run the
dashboard's A/V sync test once per camera kind so the camera's own delay is compensated (saved per input kind).

## A character folder: `characters/<name>/`
| File / field | What | Where it is used |
|---|---|---|
| `character.json` `name` | display name; also the name the persona answers to (put it in the personality) | |
| `appearance.reference` | reference photo (in this folder, in git) | JoyAI `--ref` |
| `appearance.prompt_file` | JoyAI prompt: always the preset `cloud/pipelines/joyai/prompts/replace.txt` (rule) | JoyAI `--prompt` |
| `face_pass.lpfast_preset` | preset name in `cloud/lpfast/presets.json` | `--facefix-preset` |
| `voice.*` | RVC model (`models/rvc/<name>/`, not in git: big) with `hf` = its folder in the private HF repo (`voices/<name>/<version>`, fetched by `tools/rvc_live/get_voice.py --all` / `voice_up.sh`), pitch, index rate, protect, rms_mix, block | `--voice rvc ...` |
| `conversation.persona_id` | Nuance persona that drives it (face + voice it starts from): `mira` (photo-real) works best with lpfast; `sam` is illustrated, `sonia` a 3D cartoon | Nuance session |
| `conversation.personality_file` | the persona's system prompt: who the character is, starting with "You are <name>, ..." (Nuance: the persona has no name unless the prompt gives one; spoken style, short; it must not claim to be human) | Nuance `personality.system_prompt` |
| `conversation.first_line` | what it says first (it never speaks first) | Nuance `response.create` |
| `voice.pitch_self` | pitch when YOU drive (your voice -> the character's; +5 for a male voice into Karen) | `--driver webcam/phone/srt` |
| `self_drive.test_clip` | a face + voice clip standing in for your camera/mic | `--driver webcam --test` |
| `conversation.test_clip` | a recorded session of that persona (`nuance_source.py --record`) for credit-free tests | `--test` |

## Adding a character
1. Folder + `reference.png` (head and upper body, facing the camera).
2. Voice: train an RVC v2 model on the character's speech (10+ min), put `<name>.pth` + `.index` in
   `models/rvc/<name>/`, upload the folder to the HF repo as `voices/<name>/<version>/` and set `voice.hf`;
   `cloud/infra/voice_up.sh joyai` downloads it on the JoyAI box. Pitch: 0 when
   the persona's voice is in the character's range, + semitones to go higher.
3. Face pass: record a raw JoyAI run of the test clip with the new reference (`live_bridge.py --save-input ...`,
   lpfast off), then `cloud/lpfast/sweep_preset.py` to pick gains (score: mouth size ~1x the driver, mouth corr high,
   low vertical gaze spread), save the preset in `presets.json`. See `cloud/lpfast/experiments/2026-11-25-karen/`.
4. `personality.md` + `character.json`; check with `--test`.

## Characters
| Name | Persona | Notes |
|---|---|---|
| karen | mira | first character (2026-11-25); voice v3 (Applio + TITAN 48k, epoch 275; v2 kept on HF); lpfast preset `karen` (mouth 1.08x the driver, gaze stable) |
