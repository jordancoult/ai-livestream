#!/bin/bash
# Restart the LongLive 2.0 server on the GPU box with new launch flags (box runs tools/longlive2/ll2_live.sh):
#   tools/longlive2/server_args.sh "--ckpt S4 --sage --kvq --height 704 --width 1280"
#   tools/longlive2/server_args.sh            # show the current flags
# Your client disconnects; start it again after ~1 min (warm-up + compile), with --size matching --width x --height.
cd "$(dirname "$0")/../.."
export POD_FILE="${POD_FILE:-cloud/infra/.pod.ll2}"
if [ -z "${1:-}" ]; then
  bash cloud/infra/pod.sh 'echo "running: $(tr "\0" " " < /proc/$(ss -ltnp | grep ":8765 " | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2)/cmdline | sed "s/.*ll2_server.py//")"' 2>&1 | grep -v -e Welcome -e "Have fun"
  exit 0
fi
printf '%s' "$1" | bash cloud/infra/pod.sh 'cat > /workspace/ll2_next_args; P=$(ss -ltnp | grep ":8765 " | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2); [ -n "$P" ] && kill $P; echo "restarting with: $(cat /workspace/ll2_next_args)"
for i in $(seq 1 150); do sleep 2; grep -q "ready on" /workspace/server.log 2>/dev/null && { grep "warm-up" /workspace/server.log | tail -n 1; echo READY; exit 0; }; grep -q "falling back" /workspace/ll2_live.log && { tail -n 7 /workspace/ll2_live.log; exit 1; }; done; echo "not ready after 5 min"; tail -n 5 /workspace/server.log' 2>&1 | grep -v -e Welcome -e "Have fun"
