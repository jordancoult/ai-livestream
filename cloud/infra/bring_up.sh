#!/bin/bash
# Bring up a fresh Vast 5090 for one pipeline (after vast.sh create + writing cloud/infra/.pod.<name>):
#   cloud/infra/bring_up.sh joyai     JoyAI server (fork) on :8080   -> laptop: cloud/infra/tunnel.sh start (18080)
#   cloud/infra/bring_up.sh ll        LongLive 1 + relay on :9101/9102/8090 -> laptop: tools/cloud_relay.sh
# Installs, starts the service detached and waits until it is ready. WATCHDOG=0 skips the self-stop watchdog.
set -u
cd "$(dirname "$0")/../.."
NAME=$1; export POD_FILE=cloud/infra/.pod.$NAME
P() { timeout "${2:-120}" bash cloud/infra/pod.sh "$1" 2>&1 | grep -v -e Welcome -e "Have fun"; }
put() { bash cloud/infra/pod.sh put "$1" "$2" 2>&1 | grep -v -e Welcome -e "Have fun"; }
for f in cloud/infra/vast_watchdog.sh cloud/infra/vast_heartbeat.sh; do put $f /workspace/; done
[ "${WATCHDOG:-1}" = 1 ] && P 'cd /workspace && touch .heartbeat && (setsid nohup bash vast_watchdog.sh 45 6 > watchdog.log 2>&1 < /dev/null &) && (setsid nohup bash vast_heartbeat.sh > /dev/null 2>&1 < /dev/null &); echo watchdog on'
case $NAME in
  joyai)
    for f in cloud/pipelines/joyai/setup.sh cloud/pipelines/joyai/run_server.sh cloud/pipelines/joyai/save_cache.sh; do put $f /workspace/; done
    put cloud/pipelines/joyai/prompts /workspace/
    P 'mkdir -p /root/.cache/huggingface /workspace/logs'; put .secrets/hf_token /root/.cache/huggingface/token
    P 'chmod 600 /root/.cache/huggingface/token; cd /workspace && (JOYAI_DATA=/workspace setsid nohup bash -c "bash setup.sh all && setsid nohup bash -c \"bash run_server.sh; echo EXIT=\\\$?\" > logs/server.log 2>&1 < /dev/null" > logs/setup.log 2>&1 < /dev/null &); echo setup started'
    for i in $(seq 1 120); do s=$(P 'grep -hE "Application startup complete|Traceback|EXIT=|rror:|× No solution" /workspace/logs/setup.log /workspace/logs/server.log 2>/dev/null | tail -n 2' 30); [ -n "$s" ] && { echo "$s"; break; }; sleep 20; done ;;
  ll)
    P 'cd /workspace && git clone -q https://github.com/jordancoult/ai-livestream.git ail && git clone -q -b blackwell-realtime https://github.com/jordancoult/scope.git ail/scope && cd ail && mkdir -p .tmp && (setsid nohup bash -c "bash cloud/pipelines/longlive1/setup_box.sh && HF_HUB_ENABLE_HF_TRANSFER=0 bash cloud/pipelines/longlive1/models_box.sh > .tmp/models.log 2>&1 && setsid nohup bash cloud/pipelines/longlive1/live_cloud.sh > .tmp/live_cloud.log 2>&1 < /dev/null" > .tmp/setup.out 2>&1 < /dev/null &); echo setup started'
    for i in $(seq 1 90); do s=$(P 'cd /workspace/ail; grep -h "waiting for the camera\|text-to-video\|MISSING\|Traceback" .tmp/live_cloud.log .tmp/models.log 2>/dev/null | tail -n 1' 30); [ -n "$s" ] && { echo "$s"; break; }; sleep 20; done ;;
  *) echo "usage: bring_up.sh joyai|ll"; exit 2 ;;
esac
