#!/bin/bash
# GPU-box side of the LongLive cloud relay: Scope server (5090 profile) + the bridge in a loop with the camera in on
# :9101 and the output on :9102 (laptop: tools/cloud_relay.sh). When the laptop disconnects the bridge run ends and a new
# one waits for the next connection; the server (and its loaded pipeline) stays up. "Restart with these settings" on
# the graph page (bridge exit 75 + .tmp/restart.env) is applied here: new bridge args/state, and a server restart when
# a server-side switch (compile, fp8, KV, cushion, ...) changed.
#   PRESET=20 PROMPT="..." NOISE=1.0 bash cloud/pipelines/longlive1/live_cloud.sh   (from /workspace/ail; setsid nohup)
cd "$(dirname "$0")/../../.."
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=0 MODEL=longlive NO_AUDIO=1
# what start_scope.sh exports for the LongLive family, so the graph page's launch node shows (and restarts with) them
export SCOPE_FP8_ROWWISE="${SCOPE_FP8_ROWWISE:-1}" SCOPE_COMPILE_BLOCKS="${SCOPE_COMPILE_BLOCKS:-1}" SCOPE_COMPILE_TAE="${SCOPE_COMPILE_TAE:-1}"
export SCOPE_FUSED_KV="${SCOPE_FUSED_KV:-0}" SCOPE_NOISE_T_OFFSET="${SCOPE_NOISE_T_OFFSET:-0}" SCOPE_PRELOAD_DEPTH="${SCOPE_PRELOAD_DEPTH:-1}"
export SCOPE_OUTPUT_TARGET_FRAMES="${SCOPE_OUTPUT_TARGET_FRAMES:-14}" SCOPE_SINK_QUEUE_MAX="${SCOPE_SINK_QUEUE_MAX:-24}" SCOPE_INPUT_QUEUE_MAX="${SCOPE_INPUT_QUEUE_MAX:-8}"
export PRESET="${PRESET:-20}" PROMPT="${PROMPT:-POV flying over a cyber city}" NOISE_MODE="${NOISE_MODE:-random}"
SERVER_KEYS="SCOPE_FP8_ROWWISE SCOPE_COMPILE_BLOCKS SCOPE_COMPILE_TAE SCOPE_FUSED_KV SCOPE_NOISE_T_OFFSET SCOPE_PRELOAD_DEPTH SCOPE_OUTPUT_TARGET_FRAMES SCOPE_SINK_QUEUE_MAX SCOPE_INPUT_QUEUE_MAX SCOPE_KV_FRAMES SCOPE_RING_KV SCOPE_SKIP_CLEAN_KV SCOPE_FP4_FFN SCOPE_LOCAL_ATTN_SIZE SCOPE_FRAMES_PER_BLOCK"
server_sig() { for k in $SERVER_KEYS; do printf '%s=%s;' "$k" "${!k:-}"; done; }
start_server() {
  ss -ltn | grep -q ":8000 " || (setsid nohup ./start_scope.sh --hw 5090 > .tmp/scope_server.log 2>&1 < /dev/null &)
  for i in $(seq 1 120); do curl -sf localhost:8000/api/v1/hardware/info >/dev/null && break; sleep 2; done
}
stop_server() {
  local P; P=$(ss -ltnp | grep ":8000 " | grep -o "pid=[0-9]*" | head -1 | cut -d= -f2); [ -n "$P" ] && kill "$P"
  for i in $(seq 1 30); do ss -ltn | grep -q ":8000 " || break; sleep 1; done
}
start_server
SIG=$(server_sig)
while true; do
  touch /workspace/.heartbeat
  ( source env.sh; source profiles/hw_5090.env
    REF=(); [ -n "${CHARACTER_REF:-}" ] && read -r -a REFS <<<"$CHARACTER_REF" && REF=(--ref "${REFS[@]}")
    exec scope/.venv/bin/python tools/scope_bridge.py --server http://127.0.0.1:8000 --pipeline longlive --hw 5090 \
      --preset "$PRESET" --noise-mode "$NOISE_MODE" --noise-scale "${NOISE:-1.0}" --prompt "$PROMPT" "${REF[@]}" \
      --out-size 832 480 --stream-kbps "${STREAM_KBPS:-2500}" $([ "${ORIENTATION:-portrait}" = landscape ] && echo --landscape) \
      ${BRIDGE_ARGS:-} --input "tcp://0.0.0.0:9101?listen=1" --output "tcp://0.0.0.0:9102?listen=1" )
  rc=$?
  if [ "$rc" = 75 ] && [ -f .tmp/restart.env ]; then
    echo "[live_cloud] restart with new settings: $(tr '\n' ' ' < .tmp/restart.env)"
    set -a; source .tmp/restart.env; set +a; rm -f .tmp/restart.env
    export MODEL=longlive   # this relay serves LongLive 1 only
    if [ "$(server_sig)" != "$SIG" ]; then echo "[live_cloud] server settings changed: restarting Scope"; stop_server; start_server; SIG=$(server_sig); fi
  else
    echo "[live_cloud] bridge exited ($rc); waiting for the next connection"; sleep 2
  fi
done
