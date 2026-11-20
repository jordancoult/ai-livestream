#!/bin/bash
# Run tools/bench_longlive.py on the 5090 box with start_scope.sh's LongLive env (the laptop recipe); any env overrides win.
# usage: cloud/pipelines/longlive1/bench_box.sh <tag> [bench_longlive args...]   -> RESULT line appended to /workspace/ail/.tmp/ll_results.jsonl
cd /workspace/ail && source env.sh
unset __NV_PRIME_RENDER_OFFLOAD __GLX_VENDOR_LIBRARY_NAME
export HF_HUB_ENABLE_HF_TRANSFER=0
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TORCHINDUCTOR_COMPILE_THREADS=16
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
export SCOPE_FP8_ROWWISE="${SCOPE_FP8_ROWWISE:-1}" SCOPE_COMPILE_BLOCKS="${SCOPE_COMPILE_BLOCKS:-1}" SCOPE_COMPILE_TAE="${SCOPE_COMPILE_TAE:-1}"
export SCOPE_FUSED_KV="${SCOPE_FUSED_KV:-0}" SCOPE_NOISE_T_OFFSET="${SCOPE_NOISE_T_OFFSET:-0}"
TAG=$1; shift
mkdir -p .tmp/ll
scope/.venv/bin/python tools/bench_longlive.py --tag "$TAG" "$@" > .tmp/ll/$TAG.log 2>&1
code=$?
line=$(grep -h '^RESULT ' .tmp/ll/$TAG.log | sed 's/^RESULT //')
if [ -n "$line" ]; then echo "$line" | tee -a .tmp/ll_results.jsonl; else echo "{\"tag\":\"$TAG\",\"error\":\"exit $code\"}"; tail -n 15 .tmp/ll/$TAG.log; fi
