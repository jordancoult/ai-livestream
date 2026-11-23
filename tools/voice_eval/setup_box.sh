#!/bin/bash
# Full-duplex voice model eval box (Vast, one 48 GB GPU): PersonaPlex-7B, Kyutai Moshi, NemotronLabs VoiceChat-11B,
# MiniCPM-o 4.5 (optional) + Kokoro TTS for the scripted user side. Logs in /workspace/logs/setup_*.log
set -x
W=/workspace; mkdir -p $W/logs $W/models; cd $W
export HF_HOME=$W/hf PIP_NO_CACHE_DIR=1
mkdir -p $HF_HOME && cp /root/.cache/huggingface/token $HF_HOME/token 2>/dev/null
command -v uv || (curl -LsSf https://astral.sh/uv/install.sh | sh); export PATH=$HOME/.local/bin:$PATH
apt-get update -qq && apt-get install -y -qq espeak-ng ffmpeg sox git-lfs libsndfile1 > /dev/null
TORCH="torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128"
# --- user-side TTS (Kokoro, Apache-2.0)
( uv venv -q $W/tts --python 3.12 && uv pip install -q --python $W/tts/bin/python $TORCH && \
  uv pip install -q --python $W/tts/bin/python kokoro soundfile numpy && echo TTS_OK ) > logs/setup_tts.log 2>&1 &
# --- PersonaPlex (NVIDIA fork of moshi)
( git clone -q https://github.com/NVIDIA/personaplex.git && uv venv -q $W/pp --python 3.12 && \
  uv pip install -q --python $W/pp/bin/python $TORCH && uv pip install -q --python $W/pp/bin/python ./personaplex/moshi accelerate && \
  echo PP_OK ) > logs/setup_pp.log 2>&1 &
# --- Kyutai Moshi (PyPI)
( uv venv -q $W/mo --python 3.12 && uv pip install -q --python $W/mo/bin/python $TORCH && \
  uv pip install -q --python $W/mo/bin/python moshi && echo MO_OK ) > logs/setup_mo.log 2>&1 &
# --- NemotronLabs VoiceChat (NeMo Speech branch)
( git clone -q https://github.com/NVIDIA-NeMo/Speech.git && cd Speech && git switch -q nemotron-labs-voicechat && cd .. && \
  uv venv -q $W/vc --python 3.12 && \
  uv pip install -q --python $W/vc/bin/python torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128 && \
  uv pip install -q --python $W/vc/bin/python -e "./Speech[all]" && \
  uv pip uninstall -q --python $W/vc/bin/python nvidia-resiliency-ext; \
  uv pip install -q --python $W/vc/bin/python transformers==4.56.0 tokenizers==0.22.0 lhotse==1.32.2 huggingface-hub==0.34.4 \
     hf-xet==1.1.9 torchcodec==0.10.0 torch_audiomentations jinja2 ninja packaging wheel einops setuptools && \
  MAX_JOBS=24 TORCH_CUDA_ARCH_LIST=8.9 uv pip install -q --python $W/vc/bin/python --no-build-isolation --no-deps \
     causal-conv1d==1.6.2.post1 mamba-ssm==2.3.2.post1 && echo VC_OK ) > logs/setup_vc.log 2>&1 &
# --- weights
( $W/tts/bin/python -c 1 2>/dev/null; sleep 60; \
  $W/pp/bin/python -c "from huggingface_hub import snapshot_download as s; s('nvidia/personaplex-7b-v1')" && echo W_PP_OK; \
  $W/pp/bin/python -c "from huggingface_hub import snapshot_download as s; s('kyutai/moshiko-pytorch-bf16')" && echo W_MO_OK; \
  $W/pp/bin/python -c "from huggingface_hub import snapshot_download as s; s('nvidia/NVIDIA-NemotronLabs-VoiceChat-11B', local_dir='$W/models/voicechat')" && echo W_VC_OK ) > logs/setup_weights.log 2>&1 &
wait; echo SETUP_DONE
