#!/bin/bash
# Run the JoyAI pipeline for a TikTok LIVE through OBS: bridge -> /dev/video10 (portrait 1080x1920, 25 fps) + dashboard.
#   tools/tiktok_live.sh                                  # webcam /dev/video0
#   INPUT=srt://0.0.0.0:9000 tools/tiktok_live.sh         # iPhone (Larix Broadcaster, cloud/docs/iphone-camera.md)
#   REF=assets/char_face.png PROMPT_FILE=cloud/pipelines/joyai/prompts/gstar.txt PRESET=gstar tools/tiktok_live.sh
# Needs the JoyAI tunnel: POD_FILE=cloud/infra/.pod.joyai TUNNEL_NAME=joyai cloud/infra/tunnel.sh start
# OBS setup and lip-sync offset: cloud/docs/tiktok-obs.md
cd "$(dirname "$0")/.."
INPUT=${INPUT:-/dev/video0}; REF=${REF:-assets/char_face.png}; PRESET=${PRESET:-gstar}
PROMPT_FILE=${PROMPT_FILE:-cloud/pipelines/joyai/prompts/gstar.txt}
curl -s -m 3 -o /dev/null http://127.0.0.1:18080/ || { echo "JoyAI tunnel not open (see the header of this script)"; exit 1; }
exec cloud/facefix/.venv/bin/python cloud/client/live_bridge.py --url ws://127.0.0.1:18080/ws --input "$INPUT" \
  --output /dev/video10 --out-size 1080x1920 --portrait --play-fps 25 --ref "$REF" \
  --prompt "$(cat "$PROMPT_FILE")" --facefix-preset "$PRESET" --dashboard "${DASH_PORT:-8766}" "$@"
