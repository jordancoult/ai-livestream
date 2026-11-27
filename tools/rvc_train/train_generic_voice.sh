#!/bin/bash
# Train a generic RVC voice on a rented GPU pod, with Karen's recipe (models/rvc/karen/README.txt): Applio 3.7.2
# (commit 445b0f6), TITAN-Medium 48k pretrain, 48 kHz RVC v2 + pitch, rmvpe, contentvec, Automatic cut 3.0 s / 0.3 s
# overlap / post-normalization, batch 8, fp16, 300 epochs, a checkpoint every 50.
#   on the pod:  bash train_generic_voice.sh <voice name> <VCTK speaker id>     e.g.  girl p300
# Data: VCTK 0.92 (CSTR Edinburgh, CC BY 4.0: commercial use allowed with attribution), the 48 kHz silence-trimmed
# mic1 recordings of one speaker (~20 min, like Karen's 20.7), from the HF mirror sanchit-gandhi/vctk (parquet).
# Output: /workspace/out/<name>/<name>.pth + <name>.index + checkpoints + train.log
set -euo pipefail
NAME=$1; SPK=$2
W=/workspace; A=$W/Applio; OUT=$W/out/$NAME; mkdir -p $OUT
exec > >(tee -a $OUT/train.log) 2>&1
echo "[train] $NAME from VCTK $SPK  $(date)"
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg >/dev/null)
if [ ! -d $A ]; then
  git clone -q https://github.com/IAHispano/Applio $A && git -C $A checkout -q 445b0f6
  python -m venv $W/venv && $W/venv/bin/pip install -q --upgrade pip
  $W/venv/bin/pip install -q -r $A/requirements.txt
  $W/venv/bin/pip install -q pyarrow huggingface_hub
fi
P=$W/venv/bin/python
cd $A
$P -c "from core import run_prerequisites_script; print(run_prerequisites_script(pretraineds_hifigan=False, models=True, exe=False))"
$P - <<'EOF'
from huggingface_hub import hf_hub_download
for k in ("G", "D"):
    p = hf_hub_download("blaise-tk/TITAN", f"models/medium/48k/pretrained/{k}-f048k-TITAN-Medium.pth",
                        local_dir="/workspace/titan")
    print("pretrain", p)
EOF
# --- data: the speaker's recordings from the parquet shards (filter on speaker_id; audio bytes are FLAC) ---
DS=$W/data/$NAME; mkdir -p $DS
$P - "$SPK" "$DS" <<'EOF'
import io, json, os, sys, urllib.request
import pyarrow.parquet as pq
spk, ds = sys.argv[1], sys.argv[2]
urls = json.load(urllib.request.urlopen("https://huggingface.co/api/datasets/sanchit-gandhi/vctk/parquet/default/train"))
n, sec = 0, 0.0
for u in urls:
    f = "/workspace/data/shard.parquet"
    urllib.request.urlretrieve(u, f)
    t = pq.read_table(f, columns=["speaker_id", "audio", "gender", "accent"])
    ids = t.column("speaker_id").to_pylist()
    if spk not in ids:
        continue
    for i, s in enumerate(ids):
        if s != spk:
            continue
        a = t.column("audio")[i].as_py()
        open(os.path.join(ds, f"{spk}_{n:04d}.flac"), "wb").write(a["bytes"])
        n += 1
    print(f"[data] {u.rsplit('/',1)[1]}: {n} clips so far ({t.column('gender')[ids.index(spk)]}, "
          f"{t.column('accent')[ids.index(spk)]})", flush=True)
os.remove(f)
print(f"[data] {spk}: {n} clips")
EOF
for f in $DS/*.flac; do ffmpeg -v error -y -i "$f" -ar 48000 -ac 1 "${f%.flac}.wav" && rm "$f"; done
echo "[data] $(ls $DS | wc -l) wav files, $(python3 -c "import glob,wave;print(round(sum(wave.open(f).getnframes()/48000 for f in glob.glob('$DS/*.wav'))/60,1))") min"
CORES=$(nproc)
$P - "$NAME" "$DS" "$CORES" <<'EOF'
import sys
from core import run_preprocess_script, run_extract_script, run_train_script
name, ds, cores = sys.argv[1], sys.argv[2], int(sys.argv[3])
print(run_preprocess_script(name, ds, 48000, cores, "Automatic", False, 3.0, 0.3, "post"), flush=True)
print(run_extract_script(name, "rmvpe", cores, 0, 48000, "contentvec", None, 2), flush=True)
print(run_train_script(name, 50, False, True, 300, 48000, 8, 0, True, False, "Auto", False, True,
                       "/workspace/titan/models/medium/48k/pretrained/G-f048k-TITAN-Medium.pth",
                       "/workspace/titan/models/medium/48k/pretrained/D-f048k-TITAN-Medium.pth", "HiFi-GAN", False),
      flush=True)
EOF
L=$A/logs/$NAME
ls -la $L | tail -15
cp $L/*.index $OUT/$NAME.index 2>/dev/null || true
for f in $(ls $L/${NAME}_*e_*s.pth 2>/dev/null); do cp "$f" $OUT/; done
FINAL=$(ls -v $OUT/${NAME}_*e_*s.pth | tail -1); cp "$FINAL" $OUT/$NAME.pth
echo "[train] DONE $NAME -> $OUT ($(basename $FINAL))  $(date)"
