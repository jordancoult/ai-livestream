#!/bin/bash
# JoyAI-VL-Interaction on one RTX 5090: their installer + audio runtime + models (main model quantization via $QUANT).
set -x
W=/workspace; cd $W; mkdir -p logs models
export HF_HOME=$W/hf MODEL_ROOT=$W/models HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p $HF_HOME && cp /root/.cache/huggingface/token $HF_HOME/token
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH=$HOME/.local/bin:$PATH
apt-get update -qq && apt-get install -y -qq ffmpeg git > /dev/null
[ -d jvl ] || git clone -q https://github.com/jd-opensource/JoyAI-VL-Interaction.git jvl
cd jvl
( ./install/install.sh --with-all && echo CORE_OK ) > $W/logs/install_core.log 2>&1 &
( ./install/install-audio-runtime.sh --all && echo AUDIO_OK ) > $W/logs/install_audio.log 2>&1 &
QUANT=${QUANT:-NVFP4}
( uv run --with "huggingface_hub[cli]" --no-project python - <<PY && echo MODELS_OK
from huggingface_hub import snapshot_download as s
for repo, d in [("jdopensource/JoyAI-VL-Interaction-$QUANT", "$W/models/jdopensource/JoyAI-VL-Interaction-$QUANT"),
                ("Qwen/Qwen3-VL-4B-Instruct", "$W/models/Qwen3-VL-4B-Instruct"),
                ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "$W/models/Qwen3-TTS-12Hz-1.7B-CustomVoice")]:
    s(repo, local_dir=d, max_workers=16); print("got", repo, flush=True)
PY
) > $W/logs/models.log 2>&1 &
wait; echo SETUP_DONE
