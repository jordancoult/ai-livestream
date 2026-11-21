#!/bin/bash
# Laptop side of the LongLive cloud relay: webcam -> (SSH tunnel) -> scope_bridge on the GPU box -> /dev/video10.
#   POD_FILE=cloud/infra/.pod.ll tools/cloud_relay.sh            (box: cloud/pipelines/longlive1/live_cloud.sh running)
# Tunnel: 18101 -> box 9101 (camera in), 18102 -> box 9102 (output), 18090 -> box 8090 (control page).
# Env: CAMERA=/dev/video0  CAMERA_SIZE=1280x720  UP_KBPS=1200 (your upload is the bottleneck)
#      OUT=ffplay (default: a window) | /dev/video10 (virtual camera for OBS)
#      NO_CAMERA=1: send no camera (text-only stream: pick Input -> "Text only" in the launch settings and restart)
W="$(cd "$(dirname "$0")/.." && pwd)"; cd "$W"
export POD_FILE="${POD_FILE:-cloud/infra/.pod.ll}"
TUNNEL_NAME=ll FORWARDS="18090:8090 18101:9101 18102:9102" bash cloud/infra/tunnel.sh start
for i in $(seq 1 15); do ss -ltn | grep -q ":18101 " && break; sleep 1; done
OUT="${OUT:-ffplay}"
[ "$OUT" = ffplay ] || [ -e "$OUT" ] || sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="AI_Scope_Cam" exclusive_caps=1
cleanup() { kill $UP $DOWN 2>/dev/null; wait 2>/dev/null; echo "[relay] stopped (tunnel left up: TUNNEL_NAME=ll cloud/infra/tunnel.sh stop)"; }
trap cleanup EXIT INT TERM
# camera: crop 9:16 + downscale here so only the pixels the model uses go over the upload
UP=""
[ "${NO_CAMERA:-0}" = 1 ] || ( while true; do
    if [[ "${CAMERA:-/dev/video0}" == /dev/video* ]]; then
      IN=(-f v4l2 -input_format mjpeg -video_size "${CAMERA_SIZE:-1280x720}" -framerate 30 -i "${CAMERA:-/dev/video0}")
    else IN=(-re -stream_loop -1 -i "$CAMERA"); fi   # a video file as a fake camera (tests)
    # match the box's orientation (graph page Orientation + restart): portrait = crop 9:16, landscape = full 16:9
    ORI=$(POD_FILE="$POD_FILE" timeout 10 bash cloud/infra/pod.sh 'cat /workspace/ail/.tmp/relay_orientation 2>/dev/null' 2>/dev/null | tail -n 1)
    if [ "$ORI" = landscape ]; then VF="scale=854:480"; else VF="crop=ih*9/16:ih,scale=480:854"; fi
    echo "[relay] camera -> box (${ORI:-portrait})"
    ffmpeg -loglevel error "${IN[@]}" \
      -vf "$VF" -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
      -b:v "${UP_KBPS:-1200}k" -maxrate "${UP_KBPS:-1200}k" -bufsize 500k -f mpegts tcp://127.0.0.1:18101
    sleep 1; done ) &
[ "${NO_CAMERA:-0}" = 1 ] || UP=$!
# output: box stream -> ffplay window or virtual camera, reconnecting
( while true; do
    if [ "$OUT" = ffplay ]; then
      # -timeout: give up on a dead connection after 5 s and reconnect (the box restarts its stream per session)
      ffplay -loglevel error -fflags nobuffer -flags low_delay -framedrop -probesize 32768 -analyzeduration 0 \
        -autoexit -window_title "LongLive (cloud 5090)" "tcp://127.0.0.1:18102?timeout=5000000"
    else
      ffmpeg -loglevel error -fflags nobuffer -flags low_delay -probesize 32768 -analyzeduration 0 -i "tcp://127.0.0.1:18102?timeout=5000000" \
        -pix_fmt yuv420p -f v4l2 "$OUT"
    fi
    sleep 1; done ) &
DOWN=$!
echo "[relay] webcam -> box, output -> $OUT. Controls (presets, prompt, depth): http://127.0.0.1:18090   Ctrl+C to stop"
wait $UP $DOWN 2>/dev/null
