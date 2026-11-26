#!/bin/bash
# Acceptance check of a rented GPU box BEFORE an install (~1 min): cloud/infra/box_check.sh [pod file] [profile]
# profile = the pipeline it's for: joyai (default) | ll | ll2 | voice (requirements in box_check.py PROFILES)
# Runs cloud/infra/box_check.py on it (PyTorch from the image, or the JoyAI venv if installed). exit 0 = ACCEPT.
# Rules: cloud/infra/README.md "Box acceptance". A rejected box: destroy it + add its machine_id to vast_blacklist.txt.
cd "$(dirname "$0")/../.."
export POD_FILE=${1:-cloud/infra/.pod.joyai}
bash cloud/infra/pod.sh put cloud/infra/box_check.py /workspace/ 2>&1 | grep -v -e Welcome -e "Have fun"
timeout 180 bash cloud/infra/pod.sh 'cd /workspace && P=python3; [ -x /root/venv/bin/python ] && P=/root/venv/bin/python; $P box_check.py --profile '"${2:-joyai}"' --json /workspace/box_check.json' 2>&1 | grep -v -e Welcome -e "Have fun"
