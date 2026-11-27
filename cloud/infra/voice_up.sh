#!/bin/bash
# Install the RVC voice stage on a GPU machine (default: the JoyAI box) so the bridge can run it there:
#   cloud/infra/voice_up.sh [joyai]        then: live_bridge.py ... --voice rvc --voice-host joyai
# Copies the stage code + uv + the character files, installs (idempotent, ~5 min), and downloads every character's
# voice from our private Hugging Face repo on the box (voice.hf in characters/*/character.json; needs the HF token
# that bring_up.sh puts on the box).
set -eu
cd "$(dirname "$0")/../.."
NAME=${1:-joyai}; export POD_FILE=cloud/infra/.pod.$NAME
P() { timeout "${2:-120}" bash cloud/infra/pod.sh "$1" 2>&1 | grep -v -e Welcome -e "Have fun" || true; }
P 'mkdir -p /workspace/rvc/tools/rvc_live /workspace/rvc/bin /workspace/rvc/models/rvc /workspace/rvc/cloud/third_party /workspace/rvc/characters'
bash cloud/infra/pod.sh put bin/uv /workspace/rvc/bin/
bash cloud/infra/pod.sh put tools/rvc_live/setup.sh tools/rvc_live/serve.py tools/rvc_live/rvc_engine.py tools/rvc_live/get_voice.py /workspace/rvc/tools/rvc_live/
bash cloud/infra/pod.sh put characters/* /workspace/rvc/characters/
P 'mkdir -p /workspace/rvc/voices'
bash cloud/infra/pod.sh put voices/catalog.json /workspace/rvc/voices/
P 'cd /workspace/rvc && bash tools/rvc_live/setup.sh > setup.log 2>&1; tail -1 setup.log; HF_TOKEN=$(cat /root/.cache/huggingface/token 2>/dev/null) tools/rvc_live/.venv/bin/python tools/rvc_live/get_voice.py --all' 900
