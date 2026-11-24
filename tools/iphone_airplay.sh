#!/bin/bash
# iPhone camera -> AirPlay screen mirroring -> UxPlay (this laptop) -> NVENC H.264 -> SRT -> live_bridge --input srt://...
# (instead of a phone streaming app: open a full-screen camera app on the iPhone, e.g. Blackmagic Camera, and mirror it).
#
#   1. start the bridge first:   ... live_bridge.py --input srt://0.0.0.0:9000 --portrait ...
#   2. tools/iphone_airplay.sh            (phone and laptop on the same Wi-Fi)
#      HOTSPOT=1 tools/iphone_airplay.sh  (direct laptop hotspot "UxPlay_Direct", lowest latency; asks for sudo; based on
#                                          ~/personal/phone-stream/hotspot-screen-mirror.sh)
#   3. iPhone: Control Center -> Screen Mirroring -> "JoyAI-Cam"; open the camera app full screen (portrait)
# Env: FPS (30), BITRATE kbit/s (8000), SRT_PORT (9000), ROTATE (empty = as mirrored [default]; R or L = rotate 90 deg, for a phone held in landscape)
FPS=${FPS:-30}; BITRATE=${BITRATE:-8000}; SRT_PORT=${SRT_PORT:-9000}; ROTATE=${ROTATE-}
if [ "${HOTSPOT:-0}" = 1 ]; then
  # reuse the hotspot part of the phone-stream script (everything before its uxplay line)
  S=~/personal/phone-stream/hotspot-screen-mirror.sh
  [ -f "$S" ] || { echo "missing $S"; exit 1; }
  bash <(sed '/^uxplay /,$d' "$S") || exit 1
fi
SINK="nvh264enc preset=low-latency-hq zerolatency=true bitrate=$BITRATE gop-size=$FPS ! h264parse config-interval=-1 ! mpegtsmux ! srtsink uri=srt://127.0.0.1:$SRT_PORT?mode=caller&latency=20 wait-for-connection=false"
echo "UxPlay -> SRT 127.0.0.1:$SRT_PORT ($FPS fps, $BITRATE kbit/s). Mirror the iPhone to \"JoyAI-Cam\"."
exec uxplay -n JoyAI-Cam -nh -vsync no -fps "$FPS" -reset 0 ${ROTATE:+-r $ROTATE} -vs "$SINK"
