#!/bin/bash
# Self-healing SSH tunnel: localhost:18080 -> GPU machine :8080 (reconnects whenever it drops).
#   cloud/infra/tunnel.sh start | stop | status       (address from cloud/infra/.pod)
# Several boxes/ports: POD_FILE=cloud/infra/.pod.<name> TUNNEL_NAME=<name> FORWARDS="18090:8090 18101:9101" tunnel.sh start
# (FORWARDS = local:remote port pairs, default "18080:8080"; status checks the first local port)
D="$(cd "$(dirname "$0")" && pwd)"; R="$D/../.."
PODF="$(realpath "${POD_FILE:-$D/.pod}")"; FWD="${FORWARDS:-18080:8080}"; LP1="${FWD%%:*}"
PIDF="$R/.tmp/tunnel${TUNNEL_NAME:+-$TUNNEL_NAME}.pid"; mkdir -p "$R/.tmp"
LARGS=""; for f in $FWD; do LARGS="$LARGS -L ${f%%:*}:127.0.0.1:${f##*:}"; done
case "${1:-start}" in
  start)
    if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "tunnel already running (pid $(cat "$PIDF"))"; exit 0; fi
    setsid bash -c '
      while true; do
        read -r H P < "'"$PODF"'"
        ssh -N -i "'"$R"'/.secrets/runpod_ed25519" -o UserKnownHostsFile="'"$R"'/.secrets/known_hosts" \
            -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=10 -o ServerAliveCountMax=3 \
            -o ExitOnForwardFailure=yes -o ConnectTimeout=10'"$LARGS"' -p "$P" "root@$H" 2>/dev/null
        sleep 2
      done' > /dev/null 2>&1 < /dev/null &
    echo $! > "$PIDF"; sleep 3
    ss -ltn | grep -q ":$LP1 " && echo "tunnel up ($FWD, supervisor pid $(cat "$PIDF"))" || echo "tunnel starting (supervisor pid $(cat "$PIDF"))" ;;
  stop)
    [ -f "$PIDF" ] && kill -- -"$(cat "$PIDF")" 2>/dev/null; rm -f "$PIDF"; echo "tunnel stopped" ;;
  status)
    [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null && echo "supervisor running (pid $(cat "$PIDF"))" || echo "supervisor not running"
    ss -ltn | grep -q ":$LP1 " && echo "port $LP1 listening" || echo "port $LP1 not listening" ;;
esac
