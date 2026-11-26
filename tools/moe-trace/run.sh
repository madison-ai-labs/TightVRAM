#!/usr/bin/env bash
# Trace all prompts, CPU only, alongside a running llama-server.
# Input:  prompts/*.txt + prompts/meta.json (make-prompts.py), the moe-trace binary (build.sh),
#         env MODEL (default ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf), env T (threads, default 6),
#         env PROMPTS (space-separated subset, default all), env DEADLINE (unix time; stop starting new prompts
#         when less than 60 s remain).
# Output: traces/<prompt>.bin (routing), traces/<prompt>.log
# Run:    ./run.sh
# The model is mmapped and shares the page cache with llama-server (same file); extra RAM is only KV and
# compute buffers. Timings from this run are worthless (it competes for RAM bandwidth); only routing is measured.
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
M=${MODEL:-$HOME/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}
T=${T:-6}
mkdir -p "$D/traces"
for p in ${PROMPTS:-$(python3 -c "import json;print(' '.join(json.load(open('$D/prompts/meta.json'))))")}; do
  n=$(python3 -c "import json;print(json.load(open('$D/prompts/meta.json'))['$p']['n_predict'])")
  rest=$(( ${DEADLINE:-9999999999} - $(date +%s) ))
  [ "$rest" -gt 60 ] || { echo "deadline reached, skipping $p and the rest"; break; }
  echo "== $p n=$n $(date +%T)"
  MOE_TRACE_OUT="$D/traces/$p.bin" timeout "$rest" nice -n 19 "$D/moe-trace" -m "$M" -f "$D/prompts/$p.txt" -n "$n" \
    --device none -ngl 0 -t "$T" -c 16384 -b 2048 -ub 512 -fa on > "$D/traces/$p.log" 2>&1 \
    || { echo "ERROR $p"; tail -20 "$D/traces/$p.log"; }
  grep -E "moe-trace|eval time" "$D/traces/$p.log" | tail -4
done
echo "done $(date +%T)"
