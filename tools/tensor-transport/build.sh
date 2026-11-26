#!/usr/bin/env bash
# Build the tt benchmark (tt.cpp) against a llama.cpp checkout, so it uses the same ggml libraries
# as your llama-server build (CUDA build required: libggml-cuda must exist in build/bin).
# Input:  LLAMA_SRC (env, default ~/src/llama.cpp), already built in $LLAMA_SRC/build
# Output: ./tt next to this script
# Run:    ./build.sh
set -euo pipefail
L="${LLAMA_SRC:-$HOME/src/llama.cpp}"
cd "$(dirname "$0")"
g++ -O2 -march=native -std=c++17 -o tt tt.cpp \
  -I"$L/ggml/include" -L"$L/build/bin" -Wl,-rpath,"$L/build/bin" \
  -lggml -lggml-base -lggml-cpu -lggml-cuda -lpthread
echo "built: $(pwd)/tt  (llama.cpp $(git -C "$L" rev-parse --short HEAD))"
