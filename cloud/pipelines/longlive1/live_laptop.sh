#!/bin/bash
# Laptop twin of live_box.sh: same measurement with start_scope.sh's laptop env (taskset 6-13, env.sh threads).
cd /home/jordan/personal/ai-livestream && source env.sh
export SCOPE_FP8_ROWWISE=1 SCOPE_COMPILE_BLOCKS=1 SCOPE_COMPILE_TAE=1 SCOPE_FUSED_KV=0 SCOPE_NOISE_T_OFFSET=0 SCOPE_PRELOAD_DEPTH=1
export SCOPE_OUTPUT_TARGET_FRAMES=14 SCOPE_SINK_QUEUE_MAX=24 SCOPE_INPUT_QUEUE_MAX=8 SCOPE_WEBRTC_LOCAL=1
mkdir -p .tmp/live
taskset -c 6-13 scope/.venv/bin/daydream-scope --host 127.0.0.1 --port 8000 --no-browser > .tmp/live/server_local.log 2>&1 &
SP=$!
for i in $(seq 1 90); do curl -sf localhost:8000/api/v1/hardware/info >/dev/null && break; sleep 2; done
P=(--prompt "POV flying over a cyber city" --noise-mode random --size 576 320 --preset 24 --steps 750 400)
for run in "lat 0.6" "fps 1.0"; do set -- $run
  taskset -c 6-13 scope/.venv/bin/python tools/scope_bridge.py --pipeline longlive --control-port 0 --input assets/latency_flash.mp4 \
    --output .tmp/live/local_$1.mp4 --latency-probe --duration 75 --noise-scale $2 "${P[@]}" > .tmp/live/local_$1.log 2>&1
  echo "== laptop $1"; grep "summary\|\[smooth\]" .tmp/live/local_$1.log; grep "Pipeline FPS" .tmp/live/server_local.log | tail -n 1
done
kill $SP; wait $SP 2>/dev/null
echo LOCAL_DONE
