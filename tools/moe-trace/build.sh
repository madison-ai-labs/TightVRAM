#!/usr/bin/env bash
# Build moe-trace against an existing llama.cpp build.
# Input:  env LLAMA_SRC (llama.cpp checkout, default ~/src/llama.cpp) with a finished build in $LLAMA_SRC/build.
#         Use the same commit as the server you want to compare with.
# Output: ./moe-trace next to this script (one g++ call, no cmake run over the tree).
# Run:    ./build.sh
set -euo pipefail
S=${LLAMA_SRC:-$HOME/src/llama.cpp}
B=$S/build
D=$(cd "$(dirname "$0")" && pwd)
nice -n 19 g++ -O2 -std=c++17 "$D/moe-trace.cpp" -o "$D/moe-trace" \
  -I"$S/include" -I"$S/common" -I"$S/ggml/include" -I"$S/vendor" \
  -L"$B/bin" -lllama-common -lllama -lggml -lggml-base -lpthread \
  -Wl,-rpath,"$B/bin"
echo "built: $D/moe-trace"
