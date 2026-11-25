# nuance_live: a Nuance Live AI persona as a live input (experiment, non-commercial)

`nuance_source.py` starts a Nuance Live conversation (docs.nuancelabs.ai; key in `.secrets/nuance_api_key`), joins its
LiveKit room, sends the persona's video to the bridge over SRT and its audio to a PulseAudio sink, and sends your mic
(or silence) to the persona. `tools/nuance_karen.sh` runs the whole chain:

    persona video -> SRT -> bridge -> JoyAI (face swap) -> window / OBS
    persona audio -> nuance_audio sink -> bridge --voice rvc --voice-host joyai (Karen, on the JoyAI GPU) -> delayed
                     to match the JoyAI output -> headphones
    your mic -> persona (it answers you)

    cloud/infra/tunnel.sh start          # JoyAI up
    tools/nuance_karen.sh                # live: talk to it (headphones!), dashboard on :8765
    TEST=1 tools/nuance_karen.sh         # unattended 90 s recording into ~/Videos/ai-livestream-tests

Personas: sam, sonia, mira (`PERSONA=`); video 704x640 at ~25 fps. Plan limits seen 2026-11-25: 1 conversation at a
time, 10 credits/day. The persona never speaks first: `SAY=` is its first line.

Defaults (2026-11-25): persona mira (photo-real; sam is illustrated, sonia a 3D cartoon), Karen reference
(`assets/karen-interviewer.png`) with the JoyAI preset prompt, lpfast `girl` preset, voice on the JoyAI box (run
`cloud/infra/voice_up.sh joyai` once per box). Credits: test with the recording `assets/nuance_mira_100s.mp4`
(`tools/voice_swap_test.sh`, made with `nuance_source.py --record`), not live sessions. Nuance's video carries a small
"AI" watermark that JoyAI copies into the output.
