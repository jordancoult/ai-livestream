#!/bin/bash
# For each quantization: start the main model (Triton attention, 24k context, 0.48 of the GPU), run the question and
# persona tests through the already-running adapter, stop it. Results in /workspace/runs/<quant>_*.json
W=/workspace; cd $W/jvl/services/webinfer
export HF_HOME=$W/hf PATH=$HOME/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=0
for Q in ${QUANTS:-FP8 INT8}; do
  D=$W/models/jdopensource/JoyAI-VL-Interaction-$Q
  [ -d $D ] || uv run --with "huggingface_hub[cli]" --no-project python -c "
from huggingface_hub import snapshot_download as s; s('jdopensource/JoyAI-VL-Interaction-$Q', local_dir='$D', max_workers=16)"
  VLLM_ATTENTION_BACKEND=TRITON_ATTN VLLM_USE_FLASHINFER_SAMPLER=0 MODEL_ROOT=$W/models MODEL_PATH=$D \
    SERVED_MODEL_NAME=jdopensource/JoyAI-VL-Interaction MAIN_GPU=0 MAIN_GPU_MEMORY_UTILIZATION=${UTIL:-0.48} MAX_MODEL_LEN=${MAXLEN:-24576} \
    setsid bash scripts/run.sh models > $W/logs/main_$Q.log 2>&1 < /dev/null &
  ok=0; for i in $(seq 1 60); do curl -s -m 3 localhost:7060/v1/models | grep -q '"data"' && { ok=1; break; }
    grep -aq "Engine core initialization failed" $W/logs/main_$Q.log && break; sleep 5; done
  if [ $ok = 1 ]; then
    touch $W/.heartbeat
    $W/jvl/services/.venv/bin/python $W/stream_test.py --video $W/cam30t.mp4 --out $W/runs/${Q}_questions.json --seconds 60 \
      --queries "8:Hey, can you see me? What am I wearing?;21:What am I pointing at?;31:What am I doing with my hand right now?;45:Do I look happy or surprised?" > $W/logs/${Q}_q.log 2>&1
    $W/jvl/services/.venv/bin/python $W/stream_test.py --video $W/cam30t.mp4 --out $W/runs/${Q}_persona.json --seconds 60 \
      --prompt "You are G-Star, a playful cartoon co-host on this livestream. Whenever the streamer does something expressive (a gesture, a funny face, a surprised look), react with one short, funny line. Otherwise stay silent." > $W/logs/${Q}_p.log 2>&1
    echo "$Q: ran"
  else
    echo "$Q: failed to start: $(grep -a -E 'RuntimeError:|ValueError:' $W/logs/main_$Q.log | head -1 | cut -c1-200)"
  fi
  for p in $(ps -eo pid,args | grep -E "[v]llm serve|[s]tart_all_models|[s]tart_model\.sh" | awk '{print $1}'); do kill $p; done; sleep 12
done
echo QUANT_DONE
