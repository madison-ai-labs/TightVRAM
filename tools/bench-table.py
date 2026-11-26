#!/usr/bin/env python3
"""llama-bench JSONL -> Markdown table (t/s mean +/- stddev).
Input:  one or more JSONL files from llama-sweep.sh / llama-bench -o jsonl.
Output: Markdown table on stdout.
usage: bench-table.py file.jsonl [file2.jsonl ...]
Rows: n_cpu_moe x KV type x ubatch; columns: pp512 pp4096 pp16384 tg128."""
import json, sys, collections
rows = collections.OrderedDict()
cols = []
for path in sys.argv[1:]:
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        key = (r.get("n_cpu_moe"), r["type_k"], r.get("n_ubatch"))
        test = f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}"
        if test not in cols:
            cols.append(test)
        rows.setdefault(key, {})[test] = f"{r['avg_ts']:.1f} +/- {r['stddev_ts']:.1f}"
cols.sort(key=lambda c: (c.startswith("tg"), int(c[2:])))
print("| n-cpu-moe | KV | ubatch | " + " | ".join(cols) + " |")
print("|---|---|---|" + "---|" * len(cols))
for (nc, kv, ub), v in rows.items():
    print(f"| {nc} | {kv} | {ub} | " + " | ".join(v.get(c, "OOM/-") for c in cols) + " |")
