#!/bin/bash
# usage: cloud/pipelines/longlive1/score_box.sh <out_file> <prompt> <input.mp4> clips...
cd /workspace/ail && source env.sh; unset __NV_PRIME_RENDER_OFFLOAD __GLX_VENDOR_LIBRARY_NAME; export HF_HUB_ENABLE_HF_TRANSFER=0
OUT=$1; P=$2; IN=$3; shift 3
QM_PROMPT="$P" scope/.venv/bin/python tools/quality_metrics.py "$IN" "$@" > "$OUT" 2>"$OUT.err"; echo "SCORE_EXIT=$?" >> "$OUT"
