#!/bin/bash
# Face + voice swap experiment, unattended: a talking-face clip (e.g. an AI avatar's output) is streamed as the live
# input (video over SRT, its sound into a virtual mic), JoyAI swaps the face, RVC changes the voice on the local GPU
# inside the A/V delay, and the bridge records what you would see + hear on one timeline.
#   tools/voice_swap_test.sh CLIP.mp4 [OUT.mp4]
# env: REF (reference image, default assets/blonde-gamer-girl.png), PROMPT_FILE (default replace.txt = the JoyAI preset),
#      PITCH (RVC semitones, default 0 = speaker already in the target's range; +5 for a male speaker),
#      VOICE_MODEL (default models/rvc/karen/karen.pth), URL (JoyAI ws, default tunnel), DURATION (s of clip, default 40),
#      VOICE (rvc|off), VOICE_BLOCK (ms, default 100), FACEFIX (preset name for the lpfast face pass, default none), LIVE=1 also plays it on this machine
#      (default: virtual output only, nothing audible)
set -u
cd "$(dirname "$0")/.."
CLIP=$1; OUT=${2:-test-outputs/voice-swap-test.mp4}
URL=${URL:-ws://127.0.0.1:18080/ws}; PORT=9021; SECONDS_RUN=${DURATION:-40}; T=.tmp/voiceswap; mkdir -p $T
PY=cloud/facefix/.venv/bin/python
REF=${REF:-assets/blonde-gamer-girl.png}; PROMPT=$(cat ${PROMPT_FILE:-cloud/pipelines/joyai/prompts/replace.txt})
M1=$(pactl load-module module-null-sink sink_name=clip_audio sink_properties=device.description=clip_audio)
SINK_ARGS=(); M2=""
if [ "${LIVE:-0}" != 1 ]; then
  M2=$(pactl load-module module-null-sink sink_name=voiceswap_out sink_properties=device.description=voiceswap_out)
  export PULSE_SINK=voiceswap_out
fi
cleanup(){ [ -n "${SND:-}" ] && kill $SND 2>/dev/null; pactl unload-module $M1; [ -n "$M2" ] && pactl unload-module $M2; }
trap cleanup EXIT
FF=(); [ -n "${FACEFIX:-}" ] && FF=(--facefix-preset "$FACEFIX")
timeout $((SECONDS_RUN + 120)) $PY cloud/client/live_bridge.py --url "$URL" --input "srt://0.0.0.0:$PORT" \
  --output $T/played.mp4 --portrait --ref "$REF" --prompt "$PROMPT" "${FF[@]}" \
  --audio-monitor --monitor-source clip_audio.monitor --voice ${VOICE:-rvc} --voice-pitch "${PITCH:-0}" \
  --voice-block-ms ${VOICE_BLOCK:-100} ${VOICE_SHARE:+--voice-gpu-share $VOICE_SHARE} ${VOICE_CMD:+--voice-cmd "$VOICE_CMD"} ${VOICE_HOST:+--voice-host $VOICE_HOST} --record-av-sbs \
  ${VOICE_MODEL:+--voice-model "$VOICE_MODEL"} --record-av "$OUT" --seconds "$((SECONDS_RUN + 25))" --no-reconnect \
  > $T/bridge.log 2>&1 &
BR=$!
# start the clip once the voice stage is warm (the bridge logs it), at most 90 s
for i in $(seq 1 90); do grep -q "audio monitor\|Traceback" $T/bridge.log && break; kill -0 $BR 2>/dev/null || break; sleep 1; done
sleep 3
timeout $((SECONDS_RUN + 10)) ffmpeg -v error -re -stream_loop -1 -i "$CLIP" -t "$SECONDS_RUN" -map 0:v -vf "scale=720:-2" \
  -c:v libx264 -preset veryfast -tune zerolatency -g 30 -b:v 5M -f mpegts "srt://127.0.0.1:$PORT?mode=caller&latency=20" \
  -map 0:a -ac 1 -f pulse -device clip_audio voiceswap > $T/sender.log 2>&1 &
SND=$!
wait $BR
grep -aE "voice|A/V|RESULT|Traceback|rror" $T/bridge.log | cut -c1-300 | tail -15
