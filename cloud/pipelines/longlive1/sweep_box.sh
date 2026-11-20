#!/bin/bash
# usage: cloud/pipelines/longlive1/sweep_box.sh <list> [common args...]  (detached; each line "tag args...")
cd /workspace/ail
LIST=$1; shift
while read -r tag rest; do
  [ -z "$tag" ] && continue
  eval "a=($rest)"
  bash cloud/pipelines/longlive1/bench_box.sh "$tag" "${a[@]}" "$@" --chunks 40 --out .tmp/ll/$tag.mp4 --save-input .tmp/ll/${tag}_in.mp4
done < "$LIST"
echo SWEEP_DONE >> .tmp/ll_sweep.done
