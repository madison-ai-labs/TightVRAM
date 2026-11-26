# Bytes over the wire: running Qwen3.6-35B-A3B on an 8 GB card

Patches, benchmark tools and raw results from running a 35B MoE model on an 8 GB consumer GPU.

Everything here was measured on **one machine**. The numbers tell you what happened on that box with that
build, not what will happen on yours. The tools are meant to let you measure your own.

## Hardware and software

| Part | What it is |
|---|---|
| GPU | NVIDIA RTX 3070, 8 GB GDDR6, headless |
| Link | PCIe 3.0 x16 (the CPU can't do Gen4) |
| CPU | Intel Core i9-10850K, 10 cores / 20 threads |
| RAM | 2 × 16 GB DDR4-3000 (XMP on), dual channel, single rank |
| SSD | Samsung 970 EVO (NVMe) |
| OS | Ubuntu 26.04, CUDA 12.4 |
| Model | `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (Unsloth), SHA256-checked against Hugging Face |
| llama.cpp | `45b6604b20f17b7d7638042306c0cd1b6af5999c` (build b12156) |

Build flags:

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Serving arguments used as the "production" baseline (128k context, one slot):

```sh
GGML_SCHED_PREFETCH_EXPERTS=3 llama-server -m Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  -ngl 99 --n-cpu-moe 40 -ub 4096 -fa on -t 10 -c 131072 -np 1 --load-mode dio
```

`--n-cpu-moe 40` keeps the routed experts of 40 layers in system RAM; they are uploaded to the GPU for every
prefill batch. Most of what follows is about that upload.

## Layout

```
patches/   changes to llama.cpp @45b6604b2
tools/     benchmark and analysis tools
data/      evaluations and small raw results, one directory per experiment
```

## Patches

All patches apply cleanly to llama.cpp `45b6604b2` (checked with `git apply --check`). The prefetch port only touches `ggml/src/ggml-backend.cpp`.

| File | What it does |
|---|---|
| `patch-prefetch-port.diff` | Expert prefetch: uploads the next layer's experts on a second backend instance (second CUDA stream) while the current layer computes. Enable with `GGML_SCHED_PREFETCH_EXPERTS=N` (N slots, `<=1` means 3, max 8). `GGML_SCHED_PREFETCH_DEBUG=1` prints slot size and count once. Without the variable the binary behaves like an unpatched one. |
| `port-prefetch.py` | The script that produced the diff above, by anchor-based search/replace. Useful if your checkout is a little newer: `python3 port-prefetch.py ggml/src/ggml-backend.cpp` fails loudly when an anchor doesn't match. |
| `patch-prof.diff` | Profiling build (`ggml-backend.cpp` and `src/llama-context.cpp`): timers in `compute_splits` with explicit synchronizes, split into upload and compute per graph. `LLAMA_PREFILL_PROF=1`. Measurement only, it slows things down. |
| `patch-allocdump.diff` | Enables the allocator debug output and dumps compute-buffer allocations when `GGML_ALLOC_DUMP=1`. Measurement only. |

```sh
cd llama.cpp && git checkout 45b6604b2
git apply /path/to/patches/patch-prefetch-port.diff
cmake --build build -j
```

### Credit

The prefetch logic comes from **thecodacus**, branch `prefetch-experts` at `5e7f627`
(github.com/thecodacus/llama.cpp). Upstream llama.cpp has since moved the input copy into
`ggml_backend_sched_copy_input`, so we ported it rather than cherry-picked it; the logic is unchanged.
llama.cpp is MIT-licensed, and so are the patches here.

## Tools

Server URLs and keys come from the environment: `LLAMA_URL` (default `http://127.0.0.1:8080`),
`LLAMA_API_KEY` (optional), `LLAMA_CA` (optional CA file for a TLS server with a private CA), `LLAMA_SRC` (llama.cpp checkout, default `~/src/llama.cpp`), `MODEL` (GGUF path).

| Tool | Use |
|---|---|
| `tools/llama-sweep.sh` | Runs `llama-bench` over a grid, one process per configuration, so an OOM doesn't end the run. |
| `tools/bench-table.py` | Turns `llama-bench -o jsonl` output into a Markdown table. |
| `tools/replay.py` | Load test against a running `llama-server` with real request shapes (agent turns, long documents, fixed prompts); records prefill/generation timings and VRAM. `python3 tools/replay.py --help`. |
| `tools/ram-bench.sh`, `tools/stream.c` | RAM bandwidth (STREAM plus helpers). `gcc -O3 -march=native -fopenmp stream.c -o stream`. |
| `tools/h2d.cu` | Host-to-device copy bandwidth, pinned vs. pageable. `nvcc -O2 h2d.cu -o h2d`. |
| `tools/gemm.cpp` | Dense GEMM via cuBLAS: FP16 in / FP32 accumulate, TF32, and FP32 without tensor cores as control. Checks the tensor-core rate independently of llama.cpp. `nvcc -O2 gemm.cpp -lcublas -o gemm`. |
| `tools/moe-trace/` | Records which experts every token is routed to, then simulates "hot experts on the GPU" designs. See its README. |
| `tools/tensor-transport/` | Quantizes real expert tensors into ~30 ggml formats and measures error, CPU dequant time, GPU time and upload cost. |

Each script has a header comment with inputs, outputs and how to run it.

### On other hardware

`h2d.cu`, `gemm.cpp` and the patches are CUDA/discrete-GPU specific. These are portable:

- `ram-bench.sh` + `stream.c`: any Linux box. Thread counts default to 1 / physical / logical cores
  (`THREADS_LIST` overrides); `mbw` and `dmidecode` are optional.
- `llama-sweep.sh`: any llama.cpp backend. Set `THREADS` to your physical core count; on an iGPU / UMA
  system use `--n-cpu-moe 0` configs (`"0:512 0:2048"`).
- `replay.py`: any `llama-server`. Without `nvidia-smi` the VRAM peak is recorded as `null`.

When you share numbers, include the llama.cpp commit (`llama-commit.txt` is written next to the results),
the backend and the exact serving arguments.

## Data

Directory names and most lab notes (`auswertung*.md`, `*.txt`) are in German; the numbers, JSON keys and
tables read fine without it. Server logs, traces and images are left out; what's here is the evaluation and
the per-request results it was computed from.

| Topic | Directories |
|---|---|
| A 35B model on an 8 GB card | `threads/`, `threads-batch/`, `ub-noise/`, `replay/`, `ctx-max/`, `nachtest/` |
| XMP, and the RAM bandwidth that wasn't | `xmp-on/`, `xmp-off/`, `ram-check/`, `gov-ab/` |
| Pinned memory: --load-mode none, auto and dio | `pinning/`, `loadmode/`, `stabilitaet*/` |
| Two knobs that made it slower | `loadmode/spec*`, `loadmode/moe*` |
| gpu-unlock | `gpu-unlock/` |
| Is the uncensored model better with tools? | `heretic/`, `tool-call-form/` (the "secrets" in the scenarios are fake fixtures) |
| Why a screenshot takes 12 seconds | `bild-latenz/` |
| The thinking loop that wasn't a loop | `sampling-coding-2026-11-15/`, `kontrolle-coding-2026-11-13/` |
| The simulation said +23 % | `moe-trace/`, `mtp-fork-check/` |
| Quant formats as a transport problem | `tensor-transport-20261125/` |
| Prefetching experts | `prefill-ueberlappung-20261125/` |

Glossary for file names: *auswertung* = evaluation, *lauf* = run, *roh* = raw, *kalt/warm* = cold/warm page
cache, *dauer* = duration, *stabilitaet* = stability, *nachtest* = follow-up test, *bild* = image,
*an/aus* = on/off, *reihe* = series.

## License

MIT, see `LICENSE`. The patches are derived from llama.cpp, which is also MIT.
