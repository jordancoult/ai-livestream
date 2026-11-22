#!/bin/bash
# Run ll2_server.py in a loop on the GPU box so the web client can change launch settings live:
# a {"type": "server", "args": ...} message makes the server write $NEXT and exit 75; we relaunch with those args.
# If a setting crashes the server (e.g. out of memory), we fall back to the last args that ran for 30 s.
#   setsid nohup bash /workspace/tools/longlive2/ll2_live.sh > /workspace/ll2_live.log 2>&1 &
cd /workspace
NEXT=/workspace/ll2_next_args GOOD=/workspace/ll2_good_args
[ -f "$GOOD" ] || echo "${LL2_DEFAULT_ARGS:---ckpt S2 --kvq --height 704 --width 1280}" > "$GOOD"
while true; do
  if [ -f "$NEXT" ]; then ARGS=$(cat "$NEXT"); rm -f "$NEXT"; else ARGS=$(cat "$GOOD"); fi
  touch /workspace/.heartbeat
  echo "[ll2_live] $(date -u +%T) starting: $ARGS"
  LL2_ROOT=/workspace /root/ll2/bin/python tools/longlive2/ll2_server.py $ARGS > /workspace/server.log 2>&1 &
  PID=$!
  ( for i in $(seq 1 600); do kill -0 $PID 2>/dev/null || exit 0
      grep -q "ready on" /workspace/server.log && { sleep 30; kill -0 $PID 2>/dev/null && echo "$ARGS" > "$GOOD"; exit 0; }
      sleep 1; done ) &
  wait $PID; rc=$?
  [ "$rc" = 75 ] && continue
  echo "[ll2_live] $(date -u +%T) server exited ($rc) with: $ARGS; falling back to: $(cat "$GOOD")"
  tail -n 5 /workspace/server.log
  sleep 3
done
