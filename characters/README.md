# Characters (live character mode)

A **character** is everything that makes a live persona one of ours. The live character mode
(`tools/character_live.py <name>`) turns a conversational AI persona (Nuance Live) into the character in real time:
JoyAI gives it the character's appearance, the lpfast face pass restores its expressions with the character's
tuning, RVC gives it the character's voice (in sync with the video, no added latency), and the persona talks with
the character's personality and name.

    tools/character_live.py karen            # live (headphones), output window + dashboard :8765
    tools/character_live.py karen --test     # recorded persona clip, no Nuance credits, records a review mp4

## A character folder: `characters/<name>/`
| File / field | What | Where it is used |
|---|---|---|
| `character.json` `name` | display name; also the name the persona answers to (put it in the personality) | |
| `appearance.reference` | reference photo (in this folder, in git) | JoyAI `--ref` |
| `appearance.prompt_file` | JoyAI prompt: always the preset `cloud/pipelines/joyai/prompts/replace.txt` (rule) | JoyAI `--prompt` |
| `face_pass.lpfast_preset` | preset name in `cloud/lpfast/presets.json` | `--facefix-preset` |
| `voice.*` | RVC model (`models/rvc/<name>/`, not in git: big; note its source), pitch, index rate, block | `--voice rvc ...` |
| `conversation.persona_id` | Nuance persona that drives it (face + voice it starts from): `mira` (photo-real) works best with lpfast; `sam` is illustrated, `sonia` a 3D cartoon | Nuance session |
| `conversation.personality_file` | the persona's system prompt: who the character is, starting with "You are <name>, ..." (Nuance: the persona has no name unless the prompt gives one; spoken style, short; it must not claim to be human) | Nuance `personality.system_prompt` |
| `conversation.first_line` | what it says first (it never speaks first) | Nuance `response.create` |
| `conversation.test_clip` | a recorded session of that persona (`nuance_source.py --record`) for credit-free tests | `--test` |

## Adding a character
1. Folder + `reference.png` (head and upper body, facing the camera).
2. Voice: train an RVC v2 model on the character's speech (10+ min), put `<name>.pth` + `.index` in
   `models/rvc/<name>/`, run `cloud/infra/voice_up.sh joyai` (copies all voices to the JoyAI box). Pitch: 0 when
   the persona's voice is in the character's range, + semitones to go higher.
3. Face pass: record a raw JoyAI run of the test clip with the new reference (`live_bridge.py --save-input ...`,
   lpfast off), then `cloud/lpfast/sweep_preset.py` to pick gains (score: mouth size ~1x the driver, mouth corr high,
   low vertical gaze spread), save the preset in `presets.json`. See `cloud/lpfast/experiments/2026-11-25-karen/`.
4. `personality.md` + `character.json`; check with `--test`.

## Characters
| Name | Persona | Notes |
|---|---|---|
| karen | mira | first character (2026-11-25); lpfast preset `karen` (mouth 1.08x the driver, gaze stable) |
