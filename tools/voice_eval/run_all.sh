#!/bin/bash
# Run every full-duplex model on the same scripted conversation (box: /workspace, after setup_box.sh).
# Output: /workspace/eval/<model>.wav (model voice only, same length as the input) + <model>.time (wall seconds of the
# streaming run minus model load, measured with a 2 s silent warm-up file) + <model>.log
set -u
W=/workspace; E=$W/eval; mkdir -p $E; cd $E
export HF_HOME=$W/hf
[ -f scenario.wav ] || $W/tts/bin/python $W/make_scenario.py --out scenario.wav
[ -f silence2.wav ] || sox -n -r 24000 -c 1 silence2.wav trim 0 2
PROMPT="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
timed() {  # name, then the command with IN/OUT placeholders
  local name=$1; shift
  local t0 t1 w0 w1
  t0=$(date +%s.%N); "$@" silence2.wav $name.warm.wav > $name.warm.log 2>&1; t1=$(date +%s.%N)
  w0=$(date +%s.%N); "$@" scenario.wav $name.wav > $name.log 2>&1; local rc=$?; w1=$(date +%s.%N)
  python3 -c "print(round((($w1-$w0)-($t1-$t0))/42,3))" > $name.rtf   # load cancels: (44 s run - 2 s run) / 42 s
  echo "$name rc=$rc wall=$(python3 -c "print(round($w1-$w0,1))") s, real-time factor $(cat $name.rtf) (<1 keeps up)"
}
pp() { $W/pp/bin/python -m moshi.offline --voice-prompt NATM1.pt --text-prompt "$PROMPT" --input-wav "$1" \
         --seed 42424242 --output-wav "$2" --output-text "${2%.wav}.json"; }
mo() { $W/mo/bin/python -m moshi.run_inference --batch-size 1 --hf-repo kyutai/moshiko-pytorch-bf16 "$1" "$2" && { [ -f "$2" ] || mv "${2%.wav}-0.wav" "$2"; }; }
vc() { local d; d=$(mktemp -d); $W/vc/bin/python $W/Speech/examples/speechlm2/offline_voicechat_infer.py \
         --checkpoint $W/models/voicechat --wav "$1" --output-dir "$d" && \
       f=$(find "$d" -name "*_output.wav" | head -1) && cp "$f" "$2"; ls -R "$d" > "${2%.wav}.files"; }
for m in ${MODELS:-pp mo vc}; do timed $m $m; done
echo RUN_ALL_DONE
