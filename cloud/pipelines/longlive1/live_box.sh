#!/bin/bash
# Live latency run on the box: Scope server + scope_bridge (file input with flashes), same env as start_scope.sh.
# usage: cloud/pipelines/longlive1/live_box.sh <tag> [bridge args...]   env: OUT_TARGET (cushion frames, default 14), SINK_MAX (24)
cd /workspace/ail && source env.sh
unset __NV_PRIME_RENDER_OFFLOAD __GLX_VENDOR_LIBRARY_NAME
export HF_HUB_ENABLE_HF_TRANSFER=0 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TORCHINDUCTOR_COMPILE_THREADS=16
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export SCOPE_FP8_ROWWISE="${SCOPE_FP8_ROWWISE:-1}" SCOPE_COMPILE_BLOCKS="${SCOPE_COMPILE_BLOCKS:-1}" SCOPE_COMPILE_TAE="${SCOPE_COMPILE_TAE:-1}"
export SCOPE_FUSED_KV="${SCOPE_FUSED_KV:-0}" SCOPE_NOISE_T_OFFSET="${SCOPE_NOISE_T_OFFSET:-0}" SCOPE_PRELOAD_DEPTH="${SCOPE_PRELOAD_DEPTH:-1}"
export SCOPE_OUTPUT_TARGET_FRAMES="${OUT_TARGET:-14}" SCOPE_SINK_QUEUE_MAX="${SINK_MAX:-24}" SCOPE_INPUT_QUEUE_MAX="${INPUT_MAX:-8}"
export SCOPE_WEBRTC_LOCAL=1
TAG=$1; shift
mkdir -p .tmp/live
if ! ss -ltn | grep -q ":8000 "; then
  (setsid nohup scope/.venv/bin/daydream-scope --host 127.0.0.1 --port 8000 --no-browser > .tmp/live/server.log 2>&1 < /dev/null &)
  for i in $(seq 1 90); do curl -s -o /dev/null http://127.0.0.1:8000/health && break; ss -ltn | grep -q ":8000 " && break; sleep 2; done
fi
CTRL=0; [ "${DEPTH:-0}" = 1 ] && CTRL=8090
if [ "$CTRL" != 0 ]; then  # depth is a live toggle on the control page (LongLive presets keep it off)
  ( for i in $(seq 1 120); do sleep 2; curl -sf -X POST -H "Content-Type: application/json" -d '{"depth":"on"}' http://127.0.0.1:8090/params >/dev/null && { echo "[live] depth on after ${i}x2s"; break; }; done ) &
fi
scope/.venv/bin/python tools/scope_bridge.py --pipeline longlive --control-port $CTRL --input assets/latency_flash.mp4 \
  --output .tmp/live/$TAG.mp4 --latency-probe --duration ${DURATION:-60} "$@" > .tmp/live/$TAG.log 2>&1
echo "== $TAG (cushion $SCOPE_OUTPUT_TARGET_FRAMES, depth ${DEPTH:-0})"; grep -i "summary\|\[smooth\]\|fps" .tmp/live/$TAG.log | tail -n 5
