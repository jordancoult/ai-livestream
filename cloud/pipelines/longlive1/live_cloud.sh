#!/bin/bash
# GPU-box side of the LongLive cloud relay: Scope server (5090 profile) once, then the bridge in a loop with the camera
# in on :9101 and the output on :9102 (laptop: tools/cloud_relay.sh). When the laptop disconnects the bridge run ends
# and a new one waits for the next connection; the server (and its loaded pipeline) stays up.
#   PRESET=20 PROMPT="..." NOISE=1.0 bash cloud/pipelines/longlive1/live_cloud.sh   (from /workspace/ail; setsid nohup)
cd "$(dirname "$0")/../../.."
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=0 MODEL=longlive NO_AUDIO=1
ss -ltn | grep -q ":8000 " || (setsid nohup ./start_scope.sh --hw 5090 > .tmp/scope_server.log 2>&1 < /dev/null &)
for i in $(seq 1 120); do curl -sf localhost:8000/api/v1/hardware/info >/dev/null && break; sleep 2; done
source env.sh; source profiles/hw_5090.env
while true; do
  touch /workspace/.heartbeat
  scope/.venv/bin/python tools/scope_bridge.py --server http://127.0.0.1:8000 --pipeline longlive --hw 5090 \
    --input "tcp://0.0.0.0:9101?listen=1" --output "tcp://0.0.0.0:9102?listen=1" --preset "${PRESET:-20}" \
    --noise-scale "${NOISE:-1.0}" --prompt "${PROMPT:-POV flying over a cyber city}" ${BRIDGE_ARGS:-}
  echo "[live_cloud] bridge exited ($?); waiting for the next connection"; sleep 2
done
