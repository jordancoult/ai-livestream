#!/bin/bash
# Install the RVC voice stage on a GPU machine (default: the JoyAI box) so the bridge can run it there:
#   cloud/infra/voice_up.sh [joyai]        then: live_bridge.py ... --voice rvc --voice-host joyai
# Copies tools/rvc_live + uv + the voice models (models/rvc/*) to /workspace/rvc and runs its setup (idempotent, ~5 min).
set -eu
cd "$(dirname "$0")/../.."
NAME=${1:-joyai}; export POD_FILE=cloud/infra/.pod.$NAME
P() { timeout "${2:-120}" bash cloud/infra/pod.sh "$1" 2>&1 | grep -v -e Welcome -e "Have fun" || true; }
P 'mkdir -p /workspace/rvc/tools/rvc_live /workspace/rvc/bin /workspace/rvc/models/rvc /workspace/rvc/cloud/third_party'
bash cloud/infra/pod.sh put bin/uv /workspace/rvc/bin/
bash cloud/infra/pod.sh put tools/rvc_live/setup.sh tools/rvc_live/serve.py tools/rvc_live/rvc_engine.py /workspace/rvc/tools/rvc_live/
bash cloud/infra/pod.sh put models/rvc/* /workspace/rvc/models/rvc/
P 'cd /workspace/rvc && bash tools/rvc_live/setup.sh > setup.log 2>&1; tail -1 setup.log' 900
