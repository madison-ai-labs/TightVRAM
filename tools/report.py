#!/usr/bin/env python3
"""Results directories from bench.py -> one self-contained HTML report (plus an optional Markdown summary).

    python3 tools/report.py results/<run>                     # -> results/<run>/report.html
    python3 tools/report.py results/a results/b -o cmp.html   # several runs side by side
    python3 tools/report.py results/a --no-reference          # without the RTX 3070 reference column

Input per directory (all optional except one of bench.jsonl / replay.jsonl / stream.txt):
system.json, run.json, bench.jsonl, stream.txt, replay.jsonl (see bench.py). The reference is
reference/rtx3070/ in this repo, in the same format. Standard library only; the HTML has no external
requests (no fonts, no scripts from elsewhere), so it can be mailed or attached as is.
"""
import argparse, html, json, math, re, statistics, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
E = html.escape


# --------------------------------------------------------------------------- loading

def jsonl(path):
    out = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try: out.append(json.loads(line))
                except ValueError: pass
    return out


def load(d: Path, label=None):
    j = lambda n: json.loads((d / n).read_text()) if (d / n).exists() else {}
    run = {"dir": d, "system": j("system.json"), "run": j("run.json"), "bench": jsonl(d / "bench.jsonl"),
           "replay": jsonl(d / "replay.jsonl"), "stream": []}
    if (d / "stream.txt").exists():
        for m in re.finditer(r"threads=(\d+) copy=([\d.]+) GB/s triad=([\d.]+) GB/s", (d / "stream.txt").read_text()):
            run["stream"].append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    run["label"] = label or run["run"].get("label") or d.name
    return run


def test_name(r):
    t = f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}"
    return t + (f" @ d{r['n_depth']}" if r.get("n_depth") else "")


def test_key(t):
    return (t.startswith("tg"), int(re.match(r"[a-z]+(\d+)", t).group(1)), t)


def bench_cfg(r):
    return (r.get("n_cpu_moe") or 0, r.get("n_ubatch"), r.get("type_k"), r.get("n_threads"))


# --------------------------------------------------------------------------- charts (from html/build.py of the site)

def fmt(v):
    if v is None: return "n/a"
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v) >= 100: return f"{v:.0f}"
    if float(v).is_integer(): return f"{v:.0f}"
    return f"{v:.2f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:.1f}"


def nice_max(v):
    if v <= 0: return 1
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if v <= m * mag: return m * mag
    return 10 * mag


FILLS = [("#4a2fd6", "#8b6cff"), ("#c27a12", "#ffc14d"), ("#3b6fe0", "#7fb0ff"), ("#1f9e7a", "#62e0b8"),
         ("#b03a7a", "#ff8cc6")]
REF_FILL = ("#3a3566", "#6b64a8")


def chart(title, unit, series, rows, idx, note="", ref_index=None):
    """Horizontal grouped bar chart as inline SVG plus a table view. rows: [label, v1, v2, ...]."""
    vals = [v for r in rows for v in r[1:] if v is not None]
    if not vals: return ""
    ns = len(series)
    label_w, width = 170, 720
    plot_w = width - label_w - 70
    bar_h = 15 if ns > 1 else 22
    gap_in, group_pad = 3, 14
    group_h = ns * bar_h + (ns - 1) * gap_in
    top = 8 + (22 if ns > 1 else 0)
    height = top + len(rows) * (group_h + group_pad) + 26
    vmax = nice_max(max(vals))
    sx = lambda v: plot_w * v / vmax
    fills = [REF_FILL if i == ref_index else FILLS[(i - (1 if ref_index is not None and i > ref_index else 0)) % len(FILLS)]
             for i in range(ns)]
    o = [f'<figure class="chart"><figcaption>{E(title)}</figcaption>',
         f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{E(title)}">', "<defs>"]
    o += [f'<linearGradient id="g{idx}-{i}" x1="0" x2="1"><stop offset="0" stop-color="{a}"/>'
          f'<stop offset="1" stop-color="{b}"/></linearGradient>' for i, (a, b) in enumerate(fills)]
    o.append("</defs>")
    if ns > 1:
        x = label_w
        for i, s in enumerate(series):
            o.append(f'<rect fill="url(#g{idx}-{i})" x="{x}" y="4" width="12" height="12" rx="3"/>'
                     f'<text class="legend" x="{x+18}" y="14">{E(s)}</text>')
            x += 34 + 7.5 * len(s)
    mant = round(vmax / 10 ** math.floor(math.log10(vmax)), 2)
    steps = {1.2: 4, 1.5: 3, 2: 4, 3: 3, 4: 4, 6: 3, 8: 4}.get(mant, 5)
    for i in range(steps + 1):
        v = vmax * i / steps; x = label_w + sx(v)
        o.append(f'<line class="grid{" base" if i == 0 else ""}" x1="{x:.1f}" y1="{top-4}" x2="{x:.1f}" y2="{height-22}"/>'
                 f'<text class="tick" x="{x:.1f}" y="{height-6}" text-anchor="middle">{fmt(v)}</text>')
    if unit: o.append(f'<text class="tick" x="{label_w+plot_w+30}" y="{height-6}">{E(unit)}</text>')
    y = top
    for r in rows:
        o.append(f'<text class="rowlabel" x="{label_w-12}" y="{y+group_h/2+4:.1f}" text-anchor="end">{E(str(r[0]))}</text>')
        for i, v in enumerate(r[1:]):
            by = y + i * (bar_h + gap_in)
            if v is None:
                o.append(f'<text class="value muted" x="{label_w+4}" y="{by+bar_h-3}">n/a</text>'); continue
            w = max(sx(v), 1.5); rr = min(5, w, bar_h / 2)
            path = f"M{label_w},{by} h{w-rr:.1f} q{rr},0 {rr},{rr} v{bar_h-2*rr} q0,{rr} -{rr},{rr} h-{w-rr:.1f} z"
            o.append(f'<path fill="url(#g{idx}-{i})" d="{path}"><title>{E(f"{r[0]} · {series[i]}: {fmt(v)} {unit}")}</title></path>'
                     f'<text class="value" x="{label_w+w+7:.1f}" y="{by+bar_h/2+4.5:.1f}">{fmt(v)}</text>')
        y += group_h + group_pad
    o.append("</svg>")
    if note: o.append(f'<p class="note">{E(note)}</p>')
    o.append('<details><summary>Numbers</summary><table><thead><tr><th></th>' +
             "".join(f"<th>{E(s)}</th>" for s in series) + "</tr></thead><tbody>")
    for r in rows:
        o.append(f"<tr><th>{E(str(r[0]))}</th>" + "".join(f"<td>{fmt(v)}</td>" for v in r[1:]) + "</tr>")
    o.append("</tbody></table></details></figure>")
    return "\n".join(o)


# --------------------------------------------------------------------------- sections

def sys_rows(run):
    s = run["system"]; c = s.get("cpu", {}); r = s.get("ram", {}); o = s.get("os", {}); l = s.get("llama_cpp", {})
    m = s.get("model", {}); b0 = run["bench"][0] if run["bench"] else {}
    gpus = []
    for g in s.get("gpu", []):
        if g.get("vendor") == "Vulkan":
            gpus += [f"Vulkan: {d['name']} ({d.get('type') or '?'}; {d.get('driver') or ''})" for d in g["devices"]]
        else:
            bits = [g.get("name") or g.get("vendor")]
            if g.get("vram_mib"): bits.append(f"VRAM {g['vram_mib']/1024:.1f} GiB")
            if g.get("gtt_mib"): bits.append(f"GTT {g['gtt_mib']/1024:.1f} GiB")
            if g.get("driver"): bits.append(f"driver {g['driver']}")
            if g.get("cuda_toolkit"): bits.append(f"CUDA {g['cuda_toolkit']}")
            if g.get("pcie"): bits.append(f"PCIe {g['pcie']}")
            gpus.append(", ".join(bits))
    if not gpus and b0.get("gpu_info"): gpus = [b0["gpu_info"]]
    ram = [f"{r['total_gib']} GiB" if r.get("total_gib") else None,
           " ".join(x for x in (f"{r['modules']} ×" if r.get("modules") else "", r.get("module_size"), r.get("type")) if x) or None,
           r.get("configured_speed") and f"running at {r['configured_speed']}", r.get("rank") and f"rank {r['rank']}"]
    flags = l.get("cmake") or {}
    return [
        ("CPU", c.get("model") or b0.get("cpu_info")),
        ("Cores / threads", c.get("physical") and f"{c.get('physical')} / {c.get('logical')}"),
        ("ISA", ", ".join(c.get("isa") or []) or None),
        ("Governor / EPP / boost", " / ".join(str(x) for x in (c.get("governor"), c.get("epp"), c.get("boost")) if x) or None),
        ("RAM", ", ".join(x for x in ram if x) or None),
        ("GPU", "\n".join(gpus) or None),
        ("OS", " · ".join(x for x in (o.get("distro"), o.get("kernel") and f"kernel {o['kernel']}") if x) or None),
        ("GPU-memory kernel args", o.get("kernel_args_gpu_mem")),
        ("llama.cpp", " ".join(x for x in ((l.get("commit") or b0.get("build_commit") or "")[:10], l.get("describe"),
                                         "(local changes)" if l.get("dirty") else "") if x) or None),
        ("Backend", b0.get("backends")),
        ("Build flags", " ".join(f"{k}={v}" for k, v in flags.items()) or None),
        ("Model", " · ".join(x for x in (m.get("file") or run["run"].get("model_file"), b0.get("model_type"),
                                         m.get("size_bytes") and f"{m['size_bytes']/2**30:.1f} GiB") if x) or None),
    ]


def section_system(runs):
    rows = [list(zip(*sys_rows(r))) for r in runs]
    names = rows[0][0]
    o = ['<section><h2>Hardware and software</h2><div class="scroll"><table class="sys"><thead><tr><th></th>' +
         "".join(f"<th>{E(r['label'])}</th>" for r in runs) + "</tr></thead><tbody>"]
    for i, n in enumerate(names):
        vals = [rr[1][i] for rr in rows]
        if not any(vals): continue
        o.append(f"<tr><th>{E(n)}</th>" + "".join(
            f"<td>{E(str(v)).replace(chr(10), '<br>')}</td>" if v else '<td class="muted">–</td>' for v in vals) + "</tr>")
    o.append("</tbody></table></div></section>")
    return "\n".join(o)


def best_per_test(run):
    best = {}
    for r in run["bench"]:
        t = test_name(r)
        if t not in best or r["avg_ts"] > best[t]["avg_ts"]: best[t] = r
    return best


def section_bench(runs, ref_index, counter):
    if not any(r["bench"] for r in runs): return ""
    bests = [best_per_test(r) for r in runs]
    tests = sorted({t for b in bests for t in b}, key=test_key)
    o = ['<section><h2>llama-bench</h2>',
         '<p>Best configuration per test and run (mean of the repetitions). All configurations are in the tables below.</p>']
    for kind, title in (("pp", "Prefill"), ("tg", "Generation")):
        ts = [t for t in tests if t.startswith(kind)]
        if ts:
            rows = [[t] + [b[t]["avg_ts"] if t in b else None for b in bests] for t in ts]
            o.append(chart(f"{title}, tokens per second", "t/s", [r["label"] for r in runs], rows, next(counter),
                           ref_index=ref_index))
    for run in runs:
        if not run["bench"]: continue
        cfgs = {}
        for r in run["bench"]:
            cfgs.setdefault(bench_cfg(r), {})[test_name(r)] = r
        ts = sorted({test_name(r) for r in run["bench"]}, key=test_key)
        o.append(f'<h3>{E(run["label"])}</h3><div class="scroll"><table class="num"><thead><tr>'
                 '<th>n-cpu-moe</th><th>ubatch</th><th>KV</th><th>threads</th>' + "".join(f"<th>{t}</th>" for t in ts) +
                 "</tr></thead><tbody>")
        bt = best_per_test(run)
        for (nc, ub, kv, th), v in cfgs.items():
            cells = []
            for t in ts:
                r = v.get(t)
                if not r: cells.append('<td class="muted">OOM / –</td>'); continue
                hi = ' class="best"' if bt.get(t) is r else ""
                cells.append(f"<td{hi}>{r['avg_ts']:.1f} <span>± {r['stddev_ts']:.1f}</span></td>")
            o.append(f"<tr><td>{nc}</td><td>{ub}</td><td>{kv}</td><td>{th}</td>{''.join(cells)}</tr>")
        o.append("</tbody></table></div>")
    o.append("</section>")
    return "\n".join(o)


def section_stream(runs, ref_index, counter):
    if not any(r["stream"] for r in runs): return ""
    def pick(st, which):
        if not st: return None
        one = [x for x in st if x[0] == 1]
        multi = [x for x in st if x[0] > 1]
        src = one if which == "1" else multi
        return max((x[2] for x in src), default=None)
    rows = [["Triad, 1 thread"] + [pick(r["stream"], "1") for r in runs],
            ["Triad, best of N threads"] + [pick(r["stream"], "n") for r in runs]]
    o = ['<section><h2>RAM bandwidth</h2>',
         '<p>STREAM-like triad over 3 × 1 GiB arrays (tools/stream.c, OpenMP). This is the ceiling for anything that '
         'streams expert weights from system RAM: on a UMA iGPU that is every token.</p>',
         chart("STREAM triad", "GB/s", [r["label"] for r in runs], rows, next(counter), ref_index=ref_index)]
    o.append("<details><summary>All thread counts</summary><table class=\"num\"><thead><tr><th>run</th><th>threads</th>"
             "<th>copy GB/s</th><th>triad GB/s</th></tr></thead><tbody>")
    for r in runs:
        for t, c, tr in r["stream"]:
            o.append(f"<tr><td>{E(r['label'])}</td><td>{t}</td><td>{c}</td><td>{tr}</td></tr>")
    o.append("</tbody></table></details></section>")
    return "\n".join(o)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def replay_stats(run):
    rp = run["replay"]
    it = [r for r in rp if r.get("scenario") == "interaktiv" and not r.get("error")]
    ag = [r for r in rp if r.get("scenario") == "agent" and not r.get("summary") and r.get("turn")]
    do = [r for r in rp if r.get("scenario") == "dokument"]
    sp = [r for r in rp if r.get("scenario") in ("repetitiv", "neuartig")]
    drafted = [r for r in rp if r.get("draft_n")]
    return it, ag, do, sp, drafted


def section_replay(runs, ref_index, counter):
    if not any(r["replay"] for r in runs): return ""
    stats = [replay_stats(r) for r in runs]
    labels = [r["label"] for r in runs]
    o = ['<section><h2>Replay against llama-server</h2>',
         '<p>Real request shapes, temperature 0, thinking off. <b>Interactive</b>: short question, 100–300 token '
         'answer. <b>Agent</b>: a ~15k-token prefix (system prompt, tools, source code), then tool turns with a growing '
         'history; turn 1 is cold, later turns hit the prefix cache.</p>']
    rows = [["Interactive TTFT (s)"] + [med(r["ttft"] for r in s[0]) for s in stats],
            ["Interactive total (s)"] + [med(r["wall"] for r in s[0]) for s in stats]]
    o.append(chart("Interactive, median", "s", labels, rows, next(counter), ref_index=ref_index, note="Lower is better."))
    rows = [["Interactive generation"] + [med(r["tg_tps"] for r in s[0]) for s in stats],
            ["Agent generation"] + [med(r["tg_tps"] for r in s[1]) for s in stats],
            ["Agent cold prefill"] + [med(r["pp_tps"] for r in s[1] if r.get("turn") == 1) for s in stats]]
    o.append(chart("Server throughput, median", "t/s", labels, rows, next(counter), ref_index=ref_index))
    turns = sorted({r["turn"] for s in stats for r in s[1]})
    if turns:
        def turn_wall(s, t):
            return med(r["wall"] for r in s[1] if r["turn"] == t)
        rows = [[f"Turn {t}"] + [turn_wall(s, t) for s in stats] for t in turns]
        rows.append(["Total"] + [sum(x for x in (turn_wall(s, t) for t in turns) if x) or None for s in stats])
        o.append(chart("Agent turns, wall time", "s", labels, rows, next(counter), ref_index=ref_index,
                       note="Median over passes. Lower is better."))
    for run, s in zip(runs, stats):
        it, ag, do, sp, drafted = s
        if not (ag or do or sp or drafted): continue
        o.append(f'<h3>{E(run["label"])}</h3>')
        if ag or do:
            o.append('<div class="scroll"><table class="num"><thead><tr><th>request</th><th>prompt</th><th>from cache</th>'
                     '<th>prefill t/s</th><th>generated</th><th>gen t/s</th><th>TTFT s</th><th>wall s</th><th>VRAM peak</th>'
                     '</tr></thead><tbody>')
            for r in ag + do:
                name = f"agent turn {r['turn']}" if r.get("scenario") == "agent" else f"document {r.get('ctx_target')}"
                o.append(f"<tr><td>{name}</td><td>{r.get('prompt_n') or '–'}</td><td>{r.get('cache_n') or 0}</td>"
                         f"<td>{fmt(r.get('pp_tps'))}</td><td>{r.get('predicted_n') or '–'}</td><td>{fmt(r.get('tg_tps'))}</td>"
                         f"<td>{fmt(r.get('ttft'))}</td><td>{fmt(r.get('wall'))}</td>"
                         f"<td>{str(r['vram_peak_mib']) + ' MiB' if r.get('vram_peak_mib') else '–'}</td></tr>"
                         + (f'<tr><td colspan="9" class="err">{E(r["error"])}</td></tr>' if r.get("error") else ""))
            o.append("</tbody></table></div>")
        if drafted:
            dn = sum(r["draft_n"] for r in drafted); da = sum(r.get("draft_n_accepted") or 0 for r in drafted)
            o.append(f"<p>Speculative decoding: {da} of {dn} drafted tokens accepted ({100*da/dn:.0f} %) over "
                     f"{len(drafted)} requests.</p>")
        if sp:
            o.append('<table class="num"><thead><tr><th>scenario</th><th>run</th><th>generated</th><th>gen t/s</th>'
                     '<th>drafted / accepted</th><th>wall s</th></tr></thead><tbody>')
            for r in sp:
                o.append(f"<tr><td>{'repetitive' if r['scenario']=='repetitiv' else 'novel'}</td><td>{r.get('lauf') or ''}</td>"
                         f"<td>{r.get('predicted_n')}</td><td>{fmt(r.get('tg_tps'))}</td>"
                         f"<td>{r.get('draft_n') or 0} / {r.get('draft_n_accepted') or 0}</td><td>{fmt(r.get('wall'))}</td></tr>")
            o.append("</tbody></table>")
    errs = [(r["label"], x["error"]) for r in runs for x in r["replay"] if x.get("error")]
    if errs:
        o.append(f'<p class="err">{len(errs)} requests failed, e.g. {E(errs[0][0])}: {E(errs[0][1][:200])}</p>')
    o.append("</section>")
    return "\n".join(o)


def section_commands(runs):
    o = ['<section><h2>Exact commands</h2>']
    for r in runs:
        cmds = r["run"].get("commands_shown") or []
        a = r["run"].get("args") or {}
        o.append(f'<h3>{E(r["label"])}</h3>')
        if r["run"].get("note"): o.append(f'<p class="note">{E(r["run"]["note"])}</p>')
        if r["run"].get("started"): o.append(f'<p class="note">Started {E(r["run"]["started"])}'
                                             f'{", finished " + E(r["run"]["finished"]) if r["run"].get("finished") else ""}</p>')
        if cmds:
            uniq = list(dict.fromkeys(cmds))
            o.append("<pre>" + E("\n".join(uniq)) + "</pre>")
        if a:
            o.append("<details><summary>All settings</summary><pre>" + E(json.dumps(a, indent=1)) + "</pre></details>")
    o.append("</section>")
    return "\n".join(o)


# --------------------------------------------------------------------------- summary

def summary_md(runs):
    out = []
    for r in runs:
        s = dict(sys_rows(r))
        bits = [s.get("CPU"), (s.get("GPU") or "").replace("\n", "; "), s.get("RAM"),
                " ".join(x for x in (s.get("Backend"), s.get("llama.cpp") and "llama.cpp " + s["llama.cpp"]) if x), s.get("Model")]
        out.append(f"**{r['label']}**: " + "; ".join(b for b in bits if b))
    out.append("")
    bests = [best_per_test(r) for r in runs]
    tests = sorted({t for b in bests for t in b}, key=test_key)
    if tests:
        out.append("| t/s (best config) | " + " | ".join(r["label"] for r in runs) + " |")
        out.append("|---|" + "---|" * len(runs))
        for t in tests:
            cells = []
            for b in bests:
                x = b.get(t)
                cells.append(f"{x['avg_ts']:.1f} (ub {x.get('n_ubatch')}, nc {x.get('n_cpu_moe') or 0})" if x else "–")
            out.append(f"| {t} | " + " | ".join(cells) + " |")
        out.append("")
    if any(r["stream"] for r in runs):
        out.append("| STREAM triad GB/s | " + " | ".join(r["label"] for r in runs) + " |")
        out.append("|---|" + "---|" * len(runs))
        out.append("| 1 thread | " + " | ".join(fmt(max((x[2] for x in r["stream"] if x[0] == 1), default=None)) for r in runs) + " |")
        out.append("| best N threads | " + " | ".join(fmt(max((x[2] for x in r["stream"] if x[0] > 1), default=None)) for r in runs) + " |")
        out.append("")
    if any(r["replay"] for r in runs):
        st = [replay_stats(r) for r in runs]
        out.append("| llama-server replay (median) | " + " | ".join(r["label"] for r in runs) + " |")
        out.append("|---|" + "---|" * len(runs))
        out.append("| interactive TTFT s | " + " | ".join(fmt(med(x["ttft"] for x in s[0])) for s in st) + " |")
        out.append("| interactive gen t/s | " + " | ".join(fmt(med(x["tg_tps"] for x in s[0])) for s in st) + " |")
        out.append("| agent cold prefill t/s | " + " | ".join(fmt(med(x["pp_tps"] for x in s[1] if x.get("turn") == 1)) for s in st) + " |")
        out.append("| agent gen t/s | " + " | ".join(fmt(med(x["tg_tps"] for x in s[1])) for s in st) + " |")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- page

CSS = """
:root{--ground:#08081a;--surface:#110f2a;--raised:#1a1740;--violet:#7c5cff;--lilac:#b8a6ff;--ink:#f3f1ff;--ink-2:#a7a3cc;
--ink-3:#75719c;--rule:rgba(140,116,255,.16);--warn:#ffc14d;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--ground);color:var(--ink);
font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:64rem;margin:0 auto;padding:2.5rem clamp(1rem,4vw,2.5rem) 4rem}
h1{font-size:clamp(1.7rem,4vw,2.4rem);line-height:1.15;margin:.2rem 0 .4rem;letter-spacing:-.01em}
h2{font-size:1.35rem;margin:2.8rem 0 .8rem;padding-top:1.2rem;border-top:1px solid var(--rule)}
h3{font-size:1.05rem;margin:1.6rem 0 .5rem;color:var(--lilac)}
p{color:var(--ink-2);max-width:46rem}.kicker{font:600 .8rem/1 ui-monospace,Menlo,monospace;letter-spacing:.12em;
text-transform:uppercase;color:var(--violet)}
.warn{border:1px solid rgba(255,193,77,.4);background:rgba(255,193,77,.07);color:#ffe3ad;padding:.8rem 1rem;border-radius:10px}
table{border-collapse:collapse;font-size:.88rem;font-variant-numeric:tabular-nums;width:100%}
th,td{text-align:left;padding:.4rem .7rem;border-bottom:1px solid var(--rule);vertical-align:top}
thead th{color:var(--ink-2);font-weight:600}tbody th{color:var(--ink-2);font-weight:500;white-space:nowrap}
table.sys td{min-width:14rem}table.num td span{color:var(--ink-3);font-size:.8em}td.best{color:#ffd98a;font-weight:650}
.muted{color:var(--ink-3)}.err{color:#ff9a9a}.scroll{overflow-x:auto}
figure.chart{margin:1.4rem 0;padding:1.1rem 1.2rem .8rem;background:linear-gradient(180deg,rgba(26,23,64,.72),rgba(17,15,42,.72));
border:1px solid var(--rule);border-radius:14px}
figure.chart figcaption{font-weight:650;margin-bottom:.6rem}figure.chart svg{width:100%;height:auto;display:block;overflow:visible}
svg text{font:13px system-ui,-apple-system,sans-serif;fill:var(--ink-3)}svg .rowlabel{fill:var(--ink)}
svg .value{fill:var(--ink);font-weight:650}svg .legend{fill:var(--ink-2)}svg .tick{font-size:12px}
svg .grid{stroke:rgba(140,116,255,.12)}svg .grid.base{stroke:rgba(140,116,255,.35)}
.note{font-size:.85rem;color:var(--ink-2);margin:.5rem 0 0}details{margin-top:.5rem;font-size:.88rem}
summary{cursor:pointer;color:var(--lilac)}
pre{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:.8rem 1rem;overflow-x:auto;
font:12.5px/1.5 ui-monospace,Menlo,monospace;color:var(--ink);white-space:pre-wrap;word-break:break-all}
textarea{width:100%;min-height:12rem;background:var(--surface);color:var(--ink);border:1px solid var(--rule);border-radius:10px;
padding:.8rem;font:12.5px/1.5 ui-monospace,Menlo,monospace}
footer{margin-top:3rem;color:var(--ink-3);font-size:.8rem}
"""


def page(runs, ref_index):
    counter = iter(range(1, 999))
    mine = [r for i, r in enumerate(runs) if i != ref_index]
    models = {(r["system"].get("model", {}).get("file") or r["run"].get("model_file")) for r in runs} - {None}
    warn = ""
    if ref_index is not None and len(models) > 1:
        warn = ('<p class="warn">The runs use different models. The reference is Qwen3.6-35B-A3B Q4_K_XL; compare '
                'shapes (how prefill scales with ubatch, where generation tops out) rather than absolute numbers.</p>')
    title = "llama.cpp benchmark: " + ", ".join(r["label"] for r in mine) if mine else "llama.cpp benchmark"
    head_sys = dict(sys_rows(mine[0])) if mine else {}
    sub = " · ".join(str(x).split("\n")[0] for x in (head_sys.get("CPU"), head_sys.get("GPU"), head_sys.get("RAM")) if x)
    body = [f'<p class="kicker">TightVRAM benchmark report</p><h1>{E(title)}</h1><p>{E(sub)}</p>', warn,
            section_system(runs), section_bench(runs, ref_index, counter), section_stream(runs, ref_index, counter),
            section_replay(runs, ref_index, counter), section_commands(runs),
            '<section><h2>Summary to paste</h2><p>Markdown, works on Reddit and GitHub.</p><textarea readonly>' +
            E(summary_md(runs)) + "</textarea></section>",
            f'<footer>Generated {time.strftime("%Y-%m-%d %H:%M")} by tools/report.py, '
            'github.com/madison-ai-labs/TightVRAM. Numbers are from one machine each; they say what happened on that box '
            'with that build.</footer>']
    data = [{k: v for k, v in r.items() if k != "dir"} for r in runs]
    raw = json.dumps(data, ensure_ascii=False, default=str).replace("</", "<\\/")
    return ("<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content=\"width=device-width,initial-scale=1\">"
            f"<title>{E(title)}</title><style>{CSS}</style></head><body><main>" + "\n".join(body) +
            f'</main><script type="application/json" id="raw-data">{raw}</script></body></html>')


def main():
    ap = argparse.ArgumentParser(description="bench.py results -> self-contained HTML report")
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="output HTML (default: <first dir>/report.html)")
    ap.add_argument("--summary", type=Path, help="also write the Markdown summary here")
    ap.add_argument("--labels", help="comma list, overrides the run labels")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--reference", type=Path, default=REPO / "reference" / "rtx3070")
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else []
    runs = [load(d, labels[i] if i < len(labels) else None) for i, d in enumerate(a.dirs)]
    for r in runs:
        if not (r["bench"] or r["replay"] or r["stream"]): sys.exit(f"{r['dir']}: no bench.jsonl, replay.jsonl or stream.txt")
    ref_index = None
    if not a.no_reference and a.reference.exists():
        runs.append(load(a.reference)); ref_index = len(runs) - 1
    out = a.out or a.dirs[0] / "report.html"
    out.write_text(page(runs, ref_index))
    if a.summary: a.summary.write_text(summary_md(runs) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
