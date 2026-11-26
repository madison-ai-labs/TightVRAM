#!/usr/bin/env python3
"""Tensor-transport analysis: quality x time per scenario, Pareto front.

Input:  result directory containing quant.tsv, cpu.tsv, gpu.tsv, h2d.tsv, ssd.tsv (from run-quant.sh / run-timing.sh)
Output: Markdown report on stdout
Run:    python3 -I analyze.py results/ > analysis.md

Scenarios per MoE layer of Qwen3.6-35B-A3B (gate + up + down; up has the shape of gate, so its time = gate's):
  tg-cpu       experts in RAM, CPU computes 1 token (typical llama.cpp CPU-offload setup). Extrapolated to 40 layers.
  prefill      whole layer uploaded host->VRAM (pinned) + GPU computes 4096 tokens. Serial (today) and overlapped (best case).
  tg-gpu       experts in VRAM, GPU computes 1 token (upper bound for a fixed set resident on the card).
  ssd-token    8 of 256 experts per layer per token read from NVMe (model larger than RAM), best measured rate.
Quality: activation-weighted relative error vs. BF16 (mean over 2 layers x 3 matrices).

Note on data keys: the TSV rows written by tt use some German labels which are kept as-is here:
"ausgeliefert:<type>" = shipped reference quantization, "gepinnt" = pinned host memory,
"durchgang=" = pass number prefix. Internal dict keys like "fehler" mean "error".
"""
import sys, os, statistics as st
from collections import defaultdict

D = sys.argv[1]
LAYER = 40

def kv(f):
    out = {}
    for x in f:
        if "=" in x:
            k, v = x.split("=", 1); out[k] = v
    return out

# --- quality
q = defaultdict(list); bpw = {}; qmat = defaultdict(lambda: defaultdict(list))
for z in open(os.path.join(D, "quant.tsv")):
    f = z.rstrip("\n").split("\t")
    if f[0] != "quant": continue
    t, spec, b, rel = f[1], f[3], float(f[5]), float(f[8])
    if spec.startswith("ausgeliefert:"): spec = "ausgeliefert-" + spec.split(":")[1]
    q[spec].append(rel); bpw.setdefault(spec, []).append(b)
    qmat[spec][t.split(".")[2]].append(rel)
qual = {s: st.mean(v) for s, v in q.items()}
bpwm = {s: st.mean(v) for s, v in bpw.items()}

# --- timings
def load_timing(datei, art):
    r = defaultdict(list)
    p = os.path.join(D, datei)
    if not os.path.exists(p): return r
    for z in open(p):
        f = z.rstrip("\n").split("\t")
        if f[0].startswith("durchgang="): f = f[1:]
        if f[0] != art: continue
        m = kv(f)
        ntok = int(m.get("ntok", "0"))
        r[(f[1].split(".")[2], f[2], ntok)].append(float(m["median_us"]))
    return {k: st.median(v) for k, v in r.items()}

cpu = load_timing("cpu.tsv", "cpu")
gpu = load_timing("gpu.tsv", "gpu")
h2d = {}; h2dbytes = {}
p = os.path.join(D, "h2d.tsv")
if os.path.exists(p):
    for z in open(p):
        f = z.rstrip("\n").split("\t")
        if f[0] != "h2d": continue
        m = kv(f)
        h2d[(f[1].split(".")[2], f[2], f[3])] = float(m["median_us"])
        h2dbytes[(f[1].split(".")[2], f[2])] = int(m["bytes"])
ssd_best = None
p = os.path.join(D, "ssd.tsv")
ssd_rows = []
if os.path.exists(p):
    for z in open(p):
        f = z.rstrip("\n").split("\t")
        if f[0] != "ssd": continue
        m = kv(f); ssd_rows.append(m)
        g = float(m["gbs"])
        if ssd_best is None or g > float(ssd_best["gbs"]): ssd_best = m

def layer(tab, spec, ntok):
    g, d = tab.get(("ffn_gate_exps", spec, ntok)), tab.get(("ffn_down_exps", spec, ntok))
    return None if g is None or d is None else 2 * g + d

def h2d_layer(spec, modus):
    teile = spec.split("+")
    s = 0
    for t in teile:
        g, d = h2d.get(("ffn_gate_exps", t, modus)), h2d.get(("ffn_down_exps", t, modus))
        if g is None or d is None: return None
        s += 2 * g + d
    return s

def bytes_layer(spec):
    s = 0
    for t in spec.split("+"):
        g, d = h2dbytes.get(("ffn_gate_exps", t)), h2dbytes.get(("ffn_down_exps", t))
        if g is None or d is None: return None
        s += 2 * g + d
    return s

def pareto(rows, zeit, fehler):
    rows = [r for r in rows if r[zeit] is not None]
    rows.sort(key=lambda r: (r[zeit], r[fehler]))
    best = float("inf"); front = set()
    for r in rows:
        if r[fehler] < best: best = r[fehler]; front.add(r["spec"])
    return front

specs = [s for s in qual if not s.startswith("ausgeliefert")]
rows = []
for s in specs:
    r = {"spec": s, "bpw": bpwm[s], "fehler": qual[s]}
    r["cpu_ms"] = None if layer(cpu, s, 1) is None else layer(cpu, s, 1) * LAYER / 1000
    r["gpu1_ms"] = None if layer(gpu, s, 1) is None else layer(gpu, s, 1) * LAYER / 1000
    k = layer(gpu, s, 4096); u = h2d_layer(s, "gepinnt"); up = h2d_layer(s, "pageable")
    r["pf_kern_ms"] = None if k is None else k / 1000
    r["pf_seriell_ms"] = None if k is None or u is None else (k + u) / 1000
    r["pf_ueber_ms"] = None if k is None or u is None else max(k, u) / 1000
    r["pf_pageable_ms"] = None if k is None or up is None else (k + up) / 1000
    b = bytes_layer(s)
    r["ssd_ms"] = None if b is None or ssd_best is None else b * 8 / 256 * LAYER / (float(ssd_best["gbs"]) * 1e9) * 1000
    rows.append(r)

def f(x, n=2): return "–" if x is None else f"{x:.{n}f}"

print("# Tensor transport: analysis\n")
print(f"Source: `{D}`. Error = activation-weighted relative error vs. BF16 (unsloth imatrix), mean over "
      "blk.5/blk.20 x gate/up/down. Timings from layer 20, per MoE layer = 2 x gate + down.\n")
if "ausgeliefert-q4_K" in qual:
    print(f"Reference shipped UD-Q4_K_XL (mixed Q4_K/Q5_K/Q6_K): error {qual['ausgeliefert-q4_K']:.4f}, "
          f"{bpwm['ausgeliefert-q4_K']:.2f} bpw.\n")

for titel, sp, extra in [
    ("tg on the CPU (experts in RAM, 1 token, 40 layers)", "cpu_ms", "ms/token MoE-FFN only"),
    ("tg on the GPU (experts in VRAM, 1 token, 40 layers)", "gpu1_ms", "ms/token MoE-FFN only"),
    ("Prefill ubatch 4096: pinned upload + compute, per layer", "pf_seriell_ms", "ms/layer serial"),
]:
    front = pareto(rows, sp, "fehler")
    print(f"## {titel}\n")
    if sp == "pf_seriell_ms":
        print("| Format | bpw | Error | Compute | +Upload serial | overlapped | pageable serial | pp t/s serial | pp t/s overlapped | Pareto |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|")
        for r in sorted(rows, key=lambda r: r["bpw"], reverse=True):
            ps = None if r["pf_seriell_ms"] is None else 4096 / (r["pf_seriell_ms"] * LAYER / 1000)
            pu = None if r["pf_ueber_ms"] is None else 4096 / (r["pf_ueber_ms"] * LAYER / 1000)
            print(f"| {r['spec']} | {r['bpw']:.2f} | {r['fehler']:.4f} | {f(r['pf_kern_ms'])} | {f(r['pf_seriell_ms'])} | "
                  f"{f(r['pf_ueber_ms'])} | {f(r['pf_pageable_ms'])} | {f(ps,0)} | {f(pu,0)} | {'●' if r['spec'] in front else ''} |")
        print("\n(pp t/s covers the MoE-FFN part only, without attention/Mamba, so it is an upper bound.)\n")
    else:
        print(f"| Format | bpw | Error | {extra} | Pareto |")
        print("|---|---:|---:|---:|:-:|")
        for r in sorted(rows, key=lambda r: r["bpw"], reverse=True):
            print(f"| {r['spec']} | {r['bpw']:.2f} | {r['fehler']:.4f} | {f(r[sp])} | {'●' if r['spec'] in front else ''} |")
        print()

print("## SSD (O_DIRECT)\n")
print("| Block KiB | Threads | GB/s |\n|---:|---:|---:|")
for m in ssd_rows: print(f"| {m['block_kib']} | {m['threads']} | {float(m['gbs']):.2f} |")
if ssd_best:
    print(f"\nExtrapolation per token if all experts came from the SSD (8/256 per layer, 40 layers, {float(ssd_best['gbs']):.2f} GB/s):\n")
    print("| Format | bpw | Error | ms/token SSD only |\n|---|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: r["bpw"], reverse=True):
        print(f"| {r['spec']} | {r['bpw']:.2f} | {r['fehler']:.4f} | {f(r['ssd_ms'])} |")
print("\n## Error per matrix\n")
print("| Format | gate | up | down |\n|---|---:|---:|---:|")
for s in sorted(qual, key=lambda s: -bpwm[s]):
    m = qmat[s]
    print(f"| {s} | " + " | ".join(f"{st.mean(m[k]):.4f}" if m[k] else "–" for k in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")) + " |")
