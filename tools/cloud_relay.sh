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
# Ctrl+C: stop every loop, ffmpeg and ffplay of this relay (its whole process group), so nothing keeps the camera
# (group kill only when this script leads its own group, i.e. run from a terminal; else just its descendants)
descendants() { local c; for c in $(ps -o pid= --ppid "$1"); do descendants "$c"; echo "$c"; done; }
cleanup() { trap - EXIT INT TERM; echo "[relay] stopped (tunnel left up: TUNNEL_NAME=ll cloud/infra/tunnel.sh stop)"
            if [ "$(ps -o pgid= $$ | tr -d ' ')" = "$$" ]; then kill -- -$$ 2>/dev/null
            else kill $(descendants $$) 2>/dev/null; fi; exit 0; }
trap cleanup EXIT INT TERM
# camera: crop 9:16 + downscale here so only the pixels the model uses go over the upload
UP=""
if [ "${NO_CAMERA:-0}" != 1 ]; then
( while true; do
    if [[ "${CAMERA:-/dev/video0}" == /dev/video* ]]; then
      IN=(-f v4l2 -input_format mjpeg -video_size "${CAMERA_SIZE:-1280x720}" -framerate 30 -i "${CAMERA:-/dev/video0}")
    else IN=(-re -stream_loop -1 -i "$CAMERA"); fi   # a video file as a fake camera (tests)
    # match the box's orientation (graph page Orientation + restart): portrait = crop 9:16, landscape = full 16:9
    ORI=$(POD_FILE="$POD_FILE" timeout 10 bash cloud/infra/pod.sh 'cat /workspace/ail/.tmp/relay_orientation 2>/dev/null' 2>/dev/null | tail -n 1)
    if [ "$ORI" = landscape ]; then VF="scale=854:480"; else VF="crop=ih*9/16:ih,scale=480:854"; fi
    echo "[relay] camera -> box (${ORI:-portrait})"
    # camera errors (busy device, missing camera) stay visible; the box resetting the connection does not
    ffmpeg -loglevel error "${IN[@]}" \
      -vf "$VF" -c:v libx264 -preset ultrafast -tune zerolatency -g 30 \
      -b:v "${UP_KBPS:-1200}k" -maxrate "${UP_KBPS:-1200}k" -bufsize 500k -f mpegts tcp://127.0.0.1:18101 2>&1 \
      | grep --line-buffered -v -e "Connection reset by peer" -e "Connection refused" -e "Error muxing" -e "Last message repeated" \
            -e "Error submitting a packet" -e "Error writing trailer" -e "Error closing file" -e "Terminating thread" -e "Task finished"
    echo "[relay] camera stream ended; reconnecting"
    sleep 1; done ) &
UP=$!
fi
# output: box stream -> ffplay window or virtual camera, reconnecting
( n=0; while true; do
    [ $((n % 30)) = 0 ] && echo "[relay] connecting to the box output..."; n=$((n + 1))
    if [ "$OUT" = ffplay ]; then
      # -timeout: give up on a dead connection after 5 s and reconnect (the box restarts its stream per session)
      ffplay -loglevel fatal -fflags nobuffer -flags low_delay -framedrop -probesize 32768 -analyzeduration 0 \
        -autoexit -window_title "LongLive (cloud 5090)" "tcp://127.0.0.1:18102?timeout=5000000" 2>/dev/null
    else
      ffmpeg -loglevel fatal -fflags nobuffer -flags low_delay -probesize 32768 -analyzeduration 0 -i "tcp://127.0.0.1:18102?timeout=5000000" \
        -pix_fmt yuv420p -f v4l2 "$OUT" 2>/dev/null
    fi
    sleep 1; done ) &
DOWN=$!
echo "[relay] webcam -> box, output -> $OUT. Controls (presets, prompt, depth): http://127.0.0.1:18090   Ctrl+C to stop"
wait $UP $DOWN 2>/dev/null
