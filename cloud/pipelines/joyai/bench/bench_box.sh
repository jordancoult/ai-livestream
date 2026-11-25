#!/bin/bash
# Box-local JoyAI throughput benchmark (no home-network limits). usage: [EXTRA=<start json>] bench_box.sh <tag> [fps] [seconds]
# Needs the server on :8080 and cloud/client/joyai_client.py copied to /workspace. Input is fed at <fps> (default 40,
# above what the server can do) so the result is the server ceiling; read fps_out + latency, and the server log for
# per-stage times (STREAM / STREAM-QWAIT lines).
cd /workspace; source /root/venv/bin/activate
FPS=${2:-40}; SEC=${3:-30}; EXTRA=${EXTRA:-'{"profile_timings": true, "kv_reset_frames": 0}'}
REF=JoyAI-Video-Edit/deploy/rv2v_reference/1c182f2f-32cf-4825-904e-64c69aed2e31.png
python joyai_client.py --url ws://127.0.0.1:8080/ws --video JoyAI-Video-Edit/assets/input.mp4 --ref $REF \
  --prompt "Replace me with the character in the reference picture." --fps $FPS --clip-fps $FPS --seconds $SEC \
  --codec h264 --out /root/bench_$1.mp4 --stats /root/bench_$1.json --extra "$EXTRA" 2>&1 | grep -v "^\[client\] {" | tail -4
