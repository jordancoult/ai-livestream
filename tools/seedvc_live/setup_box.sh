#!/bin/bash
# Install Seed-VC (zero-shot voice from a short clip) on a GPU box next to the JoyAI/RVC install:
#   POD_FILE=cloud/infra/.pod.joyai bash tools/seedvc_live/setup_box.sh
# Clones https://github.com/Plachtaa/seed-vc (GPL-3.0) to /workspace/seed-vc, makes its own venv with the same
# PyTorch build as RVC (2.11 cu128: works on the RTX 5090 / sm_120) + only what real-time inference needs, copies our
# headless server (tools/seedvc_live/serve.py) and warms the model cache (~25 s first load). Measured 2026-11-26 on the
# 5090: 1.6 GB VRAM, 27 ms per 180 ms block idle / 77 ms with JoyAI busy, 80 ms algorithmic delay.
set -e
cd "$(dirname "$0")/../.."
export POD_FILE=${POD_FILE:-cloud/infra/.pod.joyai}
bash cloud/infra/pod.sh "cd /workspace && (test -d seed-vc || git clone -q --depth 1 https://github.com/Plachtaa/seed-vc.git) \
  && cd seed-vc && (test -x .venv/bin/python || uv venv -q --python 3.10 .venv) \
  && uv pip install -q --python .venv/bin/python torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128 \
  && uv pip install -q --python .venv/bin/python 'numpy<2' scipy==1.13.1 librosa==0.10.2 'huggingface-hub>=0.28.1' munch==4.0.0 \
       einops==0.8.0 transformers==4.46.3 soundfile pyyaml python-dotenv tqdm hf_transfer \
  && uv pip install -q --python .venv/bin/python descript-audio-codec==1.0.0 'numpy<2' torch==2.11.0 torchaudio==2.11.0 \
       --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  && mkdir -p /workspace/seedvc_refs"
bash cloud/infra/pod.sh put tools/seedvc_live/serve.py /workspace/seed-vc/
bash cloud/infra/pod.sh put voices/girl/voice.wav /workspace/seedvc_refs/voice.wav
# warm the HF model cache (downloads the DiT, XLSR, CAMPPlus, HiFT weights once), then exit
bash cloud/infra/pod.sh "cd /workspace/seed-vc && head -c 1000000 /dev/zero | timeout 300 .venv/bin/python serve.py \
  --ref /workspace/seedvc_refs/voice.wav > /dev/null 2> /tmp/seedvc_warm.log; grep READY /tmp/seedvc_warm.log"
echo "seed-vc ready on the box"
