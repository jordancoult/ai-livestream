#!/bin/bash
# Box layout: /workspace/ail = git clone of jordancoult/ai-livestream, scope/ = jordancoult/scope (blackwell-realtime)
# LongLive/Scope on the 5090 box: venv + models (background job; logs in /workspace/ail/.tmp/)
set -u
cd /workspace/ail && mkdir -p .tmp && source env.sh
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 TORCHINDUCTOR_COMPILE_THREADS=16
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH UV_LINK_MODE=copy
( cd scope && uv sync > ../.tmp/uv_sync.log 2>&1; echo "UV_EXIT=$?" >> ../.tmp/uv_sync.log )
scope/.venv/bin/python -c "import torch, sageattention, flash_attn; print('OK', torch.__version__, torch.cuda.get_device_name(0))" >> .tmp/uv_sync.log 2>&1
for p in longlive video-depth-anything; do scope/.venv/bin/download_models -p $p > .tmp/dl_$p.log 2>&1; echo "DL_EXIT=$?" >> .tmp/dl_$p.log; done
echo SETUP_DONE >> .tmp/uv_sync.log
