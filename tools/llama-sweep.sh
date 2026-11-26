#!/bin/sh
# llama-bench sweep: (--n-cpu-moe, ubatch) pairs x KV type (K=V), -fa on, $THREADS threads, 5 repetitions.
# usage: llama-sweep.sh <label> <model.gguf> "<nc:ub> [nc:ub ...]" [extra llama-bench args]
# Input:  env LLAMA_SRC (llama.cpp checkout with build, default ~/src/llama.cpp),
#         env BENCH_DIR (output root, default ~/bench),
#         env THREADS (-t, default 10 = physical cores of the i9-10850K; set to your physical core count).
# No discrete GPU / UMA: use nc 0, e.g. "0:512 0:2048".
# Output: $BENCH_DIR/<label>/bench-<model>.jsonl (feed to bench-table.py), .log, llama-commit.txt
# Each configuration runs on its own: one OOM would otherwise abort the whole llama-bench run.
set -u
L=${1:?label}; M=${2:?model}; CFGS=${3:?configs}; shift 3
S=${LLAMA_SRC:-$HOME/src/llama.cpp}
T=${THREADS:-10}
B=$S/build/bin/llama-bench
OUT=${BENCH_DIR:-$HOME/bench}/$L; mkdir -p "$OUT"
NAME=$(basename "$M" .gguf)
J="$OUT/bench-$NAME.jsonl"; : > "$J"
( cd "$S" && git rev-parse HEAD ) > "$OUT/llama-commit.txt"
for c in $CFGS; do
  NC=${c%%:*}; UB=${c##*:}
  for KV in f16 q8_0; do
    echo "== $(date +%T) nc=$NC ub=$UB kv=$KV" >> "$OUT/bench-$NAME.log"
    "$B" -m "$M" -ngl 99 -ncmoe "$NC" -b "$UB" -ub "$UB" -fa on -t "$T" \
      -ctk $KV -ctv $KV -p 512,4096,16384 -n 128 -r 5 \
      "$@" -o jsonl >> "$J" 2>> "$OUT/bench-$NAME.log" || echo "FAIL nc=$NC ub=$UB kv=$KV" >> "$OUT/bench-$NAME.log"
  done
done
echo SWEEP_DONE >> "$OUT/bench-$NAME.log"
