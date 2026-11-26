# Our configuration, complete

Everything needed to rebuild the box the numbers come from, as of 2026-11-26. Nothing below is tuned unless it
says so; where we left a default, it is listed anyway, because defaults change between distro releases.

## Hardware

| Part | Detail |
|---|---|
| Board | ASUS ROG STRIX Z590-F GAMING WIFI, BIOS 1701 |
| CPU | Intel Core i9-10850K, 10 C / 20 T, stock clocks |
| RAM | 2 × 16 GB DDR4, XMP on (3000 MT/s), dual channel, single rank. See the XMP post for why that matters less than expected |
| GPU | RTX 3070 8 GB, VBIOS 94.04.3A.40.DB, power limit 270 W (the card's default and maximum), headless |
| Link | PCIe 3.0 x16 (Comet Lake has no Gen4) |
| SSD | Samsung 970 EVO (models and OS) |

## OS and driver

| Item | Value |
|---|---|
| OS | Ubuntu 26.04, kernel 7.0.0-43-generic |
| NVIDIA driver | 595.91.07 |
| CUDA toolkit | 12.4 (nvcc 12.4), host compiler g++-13 |
| CPU governor | `powersave` with EPP `balance_performance` (Ubuntu default) in production. `reference/rtx3070/` was measured with `performance`; data/gov-ab/ has both |
| Transparent huge pages | `madvise`, defrag `madvise` (default) |
| vm.swappiness / overcommit | 60 / 0 (defaults) |
| Kernel command line | nothing GPU- or memory-related |

## llama.cpp builds

Two builds of the same commit `45b6604b20f17b7d7638042306c0cd1b6af5999c` (b12156), both with:

```sh
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=ON \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

- **stock**: as above. Used for everything before 2026-11-25 and for `reference/rtx3070/`.
- **-pf**: plus `patches/patch-prefetch-port.diff`. Production since 2026-11-25.

## Server profiles

`profiles/*.env`, one model per file, loaded by `llama-server.service`. Comments explain each flag.

| Profile | Use | Key difference |
|---|---|---|
| `coding-pf.env` | **production** | pinned experts (`--load-mode dio`) + prefetch, 128k |
| `coding-fast.env` | long prompts < 64k | ub 8192, 64k context |
| `coding.env` | stock baseline | mmap'd experts, no prefetch; the shape of the reference numbers |
| `coding-dio.env` | stock build, pinned | +10–12 % on agent turns vs. `coding` |

Models, from `unsloth/Qwen3.6-35B-A3B-GGUF` at revision `a483e9e6cbd595906af30beda3187c2663a1118c`:

| File | SHA256 |
|---|---|
| `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | `707a55a8a4397ecde44de0c499d3e68c1ad1d240d1da65826b4949d1043f4450` |
| `mmproj-F16.gguf` (saved as `Qwen3.6-35B-A3B-mmproj-F16.gguf`) | `8971ee4f331ff0a4c609374f32984b3d4e6dc086c0aa35f1d637fad1829e887f` |

`python3 tools/bench.py --hash` writes the SHA256 of your file into the report.

## Sampling

The benchmarks send their own sampling (temperature 0, fixed seed, thinking off), so the server defaults
don't affect them. For agent use, the sampling post decided on the model card's coding sampler:

```
--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
```

All profiles above carry these flags, as the live profiles on the box do since 2026-11-26. They are only the
server default: a client that sends its own sampling fields overrides them.

## Reproducing the reference numbers

```sh
python3 tools/bench.py --preset rtx3070 --model /path/to/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --llama /path/to/llama.cpp --replay --scenarios interactive,agent,document
```

`--preset rtx3070` sets the exact llama-bench and llama-server arguments of `reference/rtx3070/`
(stock build, `-ncmoe 40 -ub 4096 -b 4096 -t 10 -fa on`, KV f16, pp512/4096/16384 + tg128, 5 repetitions, 128k
server context). For the production setup on top of that, use the -pf build and add
`GGML_SCHED_PREFETCH_EXPERTS=3` in front and `--server-extra "--load-mode dio"`.
