#!/bin/bash
# Download the LongLive 1 model set (what longlive/schema.py declares) + depth + CLIP (quality_metrics) on the box.
set -u
cd /workspace/ail && source env.sh
H="scope/.venv/bin/hf"; [ -x $H ] || H="scope/.venv/bin/huggingface-cli"
M=models
$H download daydreamlive/Wan2.1-T2V-1.3B config.json Wan2.1_VAE.pth --include "google/*" --local-dir $M/Wan2.1-T2V-1.3B >/dev/null 2>&1 || \
  $H download daydreamlive/Wan2.1-T2V-1.3B --include "config.json" "Wan2.1_VAE.pth" "google/*" --local-dir $M/Wan2.1-T2V-1.3B
$H download daydreamlive/WanVideo_comfy --include "config.json" "umt5-xxl-enc-fp8_e4m3fn.safetensors" "Wan2_1-VACE_module_1_3B_bf16.safetensors" --local-dir $M/WanVideo_comfy
$H download daydreamlive/Autoencoders --include "config.json" "lighttaew2_1.pth" "lightvaew2_1.pth" "taew2_1.pth" --local-dir $M/Autoencoders
$H download daydreamlive/LongLive-1.3B --include "config.json" "models/longlive_base.pt" "models/lora.pt" --local-dir $M/LongLive-1.3B
$H download daydreamlive/Video-Depth-Anything-Small --local-dir $M/Video-Depth-Anything-Small
$H download openai/clip-vit-large-patch14 --include "*.json" "*.txt" "model.safetensors" >/dev/null
du -sh $M/*
echo MODELS_DONE
