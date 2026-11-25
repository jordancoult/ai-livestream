#!/bin/bash
# Nuance Live persona -> JoyAI face swap + RVC voice (Karen), audio delayed in sync with the JoyAI output.
# Non-commercial experiment. JoyAI tunnel must be up (cloud/infra/tunnel.sh start).
#   tools/nuance_karen.sh                 live: you talk to the persona with your mic (HEADPHONES), output in a window
#   TEST=1 tools/nuance_karen.sh          unattended: no mic (persona just talks), nothing audible, records the result
# env: PERSONA (sam|sonia|mira, default sonia), PITCH (RVC semitones, default 0: the personas are in Karen's range;
#      try +2..+5 for sam), REF / PROMPT_FILE (JoyAI target, default the blonde reference), MAX_S (session length,
#      default 300; TEST default 90), SAY (first line the persona says), OUT (record mp4, default in the review folder),
#      FACEFIX (lpfast preset, e.g. girl), DASH (dashboard port, live only, default 8765), MIC (Pulse source, default)
set -u
cd "$(dirname "$0")/.."
URL=${URL:-ws://127.0.0.1:18080/ws}; PORT=9022; T=.tmp/nuance; mkdir -p $T
PY=cloud/facefix/.venv/bin/python; NPY=tools/nuance_live/.venv/bin/python
REF=${REF:-assets/blonde-gamer-girl.png}; PROMPT=$(cat ${PROMPT_FILE:-cloud/pipelines/joyai/prompts/human_blonde.txt})
TEST=${TEST:-0}; MAX_S=${MAX_S:-$([ "$TEST" = 1 ] && echo 90 || echo 300)}
OUT=${OUT:-$HOME/Videos/ai-livestream-tests/nuance-karen-$(date +%H%M).mp4}
[ -s .secrets/nuance_api_key ] || { echo "missing .secrets/nuance_api_key"; exit 1; }
[ -x $NPY ] || { bin/uv venv -q -p 3.12 tools/nuance_live/.venv && bin/uv pip install -q -p $NPY livekit aiohttp numpy; }
M1=$(pactl load-module module-null-sink sink_name=nuance_audio sink_properties=device.description=nuance_audio)
M2=""
if [ "$TEST" = 1 ]; then
  M2=$(pactl load-module module-null-sink sink_name=nuance_out sink_properties=device.description=nuance_out)
  export PULSE_SINK=nuance_out
  OUTPUT=(--output $T/played.mp4); MICARG=none; SAY=${SAY:-"Say hi to your viewers, introduce yourself, and tell them a short funny story about your week. Keep talking for about a minute."}
else
  OUTPUT=(--output ffplay --dashboard ${DASH:-8765}); MICARG=${MIC:-default}; SAY=${SAY:-"Say hi to your viewers and introduce yourself."}
fi
cleanup(){ [ -n "${NC:-}" ] && kill -TERM $NC 2>/dev/null && sleep 3; [ -n "${BR:-}" ] && kill -TERM $BR 2>/dev/null; sleep 1
           pactl unload-module $M1; [ -n "$M2" ] && pactl unload-module $M2; }
trap cleanup EXIT
FF=(); [ -n "${FACEFIX:-}" ] && FF=(--facefix-preset "$FACEFIX")
$PY cloud/client/live_bridge.py --url "$URL" --input "srt://0.0.0.0:$PORT" "${OUTPUT[@]}" --portrait --ref "$REF" \
  --prompt "$PROMPT" "${FF[@]}" --audio-monitor --monitor-source nuance_audio.monitor --voice rvc \
  --voice-pitch "${PITCH:-0}" --record-av "$OUT" --record-av-sbs --seconds $((MAX_S + 40)) > $T/bridge.log 2>&1 &
BR=$!
for i in $(seq 1 90); do grep -q "audio monitor\|Traceback" $T/bridge.log && break; kill -0 $BR 2>/dev/null || break; sleep 1; done
grep -a "voice\|Traceback" $T/bridge.log
$NPY tools/nuance_live/nuance_source.py --srt "srt://127.0.0.1:$PORT" --sink nuance_audio --persona "${PERSONA:-sonia}" \
  --mic "$MICARG" --say "$SAY" --max-seconds "$MAX_S" > >(grep --line-buffered -v "^libwebrtc" | tee $T/nuance.log) 2>&1 &
NC=$!
wait $NC; NC=""
sleep 3; kill -TERM $BR 2>/dev/null; wait $BR; BR=""
grep -aE "A/V recording|RESULT|Traceback" $T/bridge.log | cut -c1-400
