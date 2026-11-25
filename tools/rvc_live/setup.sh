#!/bin/bash
# Realtime RVC voice stage on the local GPU (Applio's realtime core, MIT): venv + Applio (pinned) + its model assets.
#   tools/rvc_live/setup.sh          then put a voice model in models/rvc/<name>/ (<name>.pth + .index)
set -eu
cd "$(dirname "$0")/../.."
APPLIO=cloud/third_party/applio; REV=445b0f628ad2481415871325b7012d01cc9b52bc
if [ ! -d $APPLIO/.git ]; then
  git init -q $APPLIO && git -C $APPLIO remote add origin https://github.com/IAHispano/Applio.git
  git -C $APPLIO fetch -q --depth 1 origin $REV && git -C $APPLIO checkout -q FETCH_HEAD
fi
[ -x tools/rvc_live/.venv/bin/python ] || bin/uv venv -q -p 3.12 tools/rvc_live/.venv
bin/uv pip install -q -p tools/rvc_live/.venv/bin/python torch==2.11.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
grep -v -E "^torch(audio)?==|win32|edge-tts|tensorboard|matplotlib" $APPLIO/requirements-engine.txt \
  | sed 's/; sys_platform != "win32"//' > tools/rvc_live/.venv/req.txt
echo 'onnxruntime>=1.17.0' >> tools/rvc_live/.venv/req.txt
bin/uv pip install -q -p tools/rvc_live/.venv/bin/python -r tools/rvc_live/.venv/req.txt
B=https://huggingface.co/IAHispano/Applio/resolve/main/Resources
mkdir -p $APPLIO/rvc/models/predictors $APPLIO/rvc/models/embedders/contentvec
for f in predictors/rmvpe.pt predictors/fcpe.pt embedders/contentvec/pytorch_model.bin embedders/contentvec/config.json; do
  [ -s $APPLIO/rvc/models/$f ] || curl -sSfL -o $APPLIO/rvc/models/$f $B/$f
done
tools/rvc_live/.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print('rvc_live ready:', torch.cuda.get_device_name())"
