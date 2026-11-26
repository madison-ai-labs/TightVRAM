#!/usr/bin/env python3
"""Replay load test for llama-server (run it on the test machine or anywhere that can reach the server).

Scenarios (the scenario names below are CLI values and appear as "scenario" in the JSONL; kept as-is):
  interaktiv  short prompt, 100-300 token answer -> TTFT, total time            (interactive chat)
  agent       10-20k token prefix (system + tools + source code), several tool turns with a growing
              history -> time per turn, cache_n vs. prompt_n (timings), total time
  dokument    32k / 64k token document + question, cold -> prefill time, VRAM peak, stability   (document)
  repetitiv   answer repeats the prompt almost verbatim (best case for n-gram speculative decoding)
  neuartig    free-form generation without repetition (worst case for n-gram speculation)

Every request writes one JSON line to --out (raw data); a summary is printed to stdout at the end.
Some JSON keys/values are German and kept for compatibility with existing data files: "art" = kind
(json|code), "lauf" = run number, "dokument"/"interaktiv" = scenario names, "kalt" = cold.
Determinism: temperature 0, fixed seed, thinking off (--thinking turns it on). The history is only appended
to, never rewritten (otherwise the prefix cache does not apply). "Cold" means a unique nonce at the start of
the system prompt so that no prefix can match the cache.
Note: the prompts are German on purpose (as measured); translating them would change token counts.

Environment:
  LLAMA_URL      server base URL (default http://127.0.0.1:8080)
  LLAMA_API_KEY  optional; sent as "Authorization: Bearer ..." only if set
  LLAMA_CA       optional path to a CA bundle for an https server with a private CA
  LLAMA_SRC      llama.cpp checkout whose source files are used as prompt material (default ~/src/llama.cpp)
Uses nvidia-smi (if on PATH) for the VRAM peak of the "dokument" scenario; without it the peak is null.

usage: replay.py [interaktiv|agent|dokument|alle|repetitiv|neuartig] [--out f.jsonl] [--label x] [--runs n] [--loop sec]
Default output: ./bench/<label>/replay-HHMMSS.jsonl next to this script.
"""
import argparse, hashlib, json, os, random, ssl, statistics, subprocess, sys, threading, time, urllib.request, uuid, glob

URL = os.environ.get("LLAMA_URL", "http://127.0.0.1:8080")
KEY = os.environ.get("LLAMA_API_KEY", "")
CA = os.environ.get("LLAMA_CA")
SRC = os.environ.get("LLAMA_SRC") or os.path.expanduser("~/src/llama.cpp")
CTX = ssl.create_default_context(cafile=CA) if CA else None


def http(path, body=None, timeout=1800):
    headers = {"Content-Type": "application/json"}
    if KEY: headers["Authorization"] = "Bearer " + KEY
    r = urllib.request.Request(URL + path, json.dumps(body).encode() if body is not None else None, headers)
    return urllib.request.urlopen(r, timeout=timeout, context=CTX)


def ntok(text):
    return len(json.load(http("/tokenize", {"content": text}))["tokens"])


class VramSampler(threading.Thread):
    """Polls nvidia-smi every 100 ms and remembers the peak (MiB) during a request."""
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0; self.stop = threading.Event()
        try:
            self.p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                                       "-lms", "100"], stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            self.p = None  # no NVIDIA GPU (e.g. AMD/Vulkan, Apple): vram_peak_mib is recorded as null
    def run(self):
        if self.p is None: return
        for line in self.p.stdout:
            try: self.peak = max(self.peak, int(line.strip()))
            except ValueError: pass
            if self.stop.is_set(): break
    def done(self):
        self.stop.set()
        if self.p is None: return None
        self.p.terminate(); return self.peak


def chat(messages, max_tokens, tools=None, thinking=False, vram=False):
    """Streams one chat request. Returns a dict with ttft, wall, timings, text, tool calls."""
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 1,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if tools: body["tools"] = tools
    vs = VramSampler() if vram else None
    if vs: vs.start()
    t0 = time.time(); ttft = None; text = ""; timings = {}; calls = {}; finish = None; err = None
    try:
        with http("/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]": continue
                ev = json.loads(line[6:])
                if "timings" in ev: timings = ev["timings"]
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    if (d.get("content") or d.get("tool_calls") or d.get("reasoning_content")) and ttft is None:
                        ttft = time.time() - t0
                    text += d.get("content") or ""
                    for tc in d.get("tool_calls") or []:
                        c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        c["id"] = tc.get("id") or c["id"]
                        f = tc.get("function", {})
                        c["name"] += f.get("name") or ""; c["arguments"] += f.get("arguments") or ""
                    finish = ch.get("finish_reason") or finish
    except Exception as e:  # stability test: log errors instead of aborting
        err = f"{type(e).__name__}: {e}"
    res = {"wall": time.time() - t0, "ttft": ttft, "finish": finish, "error": err, "text": text,
           "tool_calls": [calls[k] for k in sorted(calls)],
           "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"),
           "prompt_ms": timings.get("prompt_ms"), "predicted_n": timings.get("predicted_n"),
           "predicted_ms": timings.get("predicted_ms"), "pp_tps": timings.get("prompt_per_second"),
           "tg_tps": timings.get("predicted_per_second"),
           "draft_n": timings.get("draft_n"), "draft_n_accepted": timings.get("draft_n_accepted"),
           "text_sha": hashlib.sha256(text.encode()).hexdigest()[:12]}
    res["tool_ok"] = all(tool_json_ok(c["arguments"]) for c in res["tool_calls"])
    if vs: res["vram_peak_mib"] = vs.done()
    return res


def tool_json_ok(arguments):
    try:
        return isinstance(json.loads(arguments), dict)
    except ValueError:
        return False


def corpus(min_tokens, pattern):
    """Real source files from the llama.cpp checkout, sorted, until min_tokens is reached (roughly by characters, trimmed exactly later)."""
    parts = []; chars = 0
    for f in sorted(glob.glob(os.path.join(SRC, pattern), recursive=True)):
        try: s = open(f, errors="replace").read()
        except OSError: continue
        parts.append(f"=== {os.path.relpath(f, SRC)} ===\n{s}\n"); chars += len(s)
        if chars > min_tokens * 3.2: break
    return "".join(parts)


def trim_to(text, tokens):
    """Trim text to ~tokens tokens (binary search over the character length using /tokenize)."""
    lo, hi = 0, len(text)
    while hi - lo > 2000:
        mid = (lo + hi) // 2
        if ntok(text[:mid]) < tokens: lo = mid
        else: hi = mid
    return text[:lo]


TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "grep", "description": "Search the repository with a regex",
     "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                    "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace a string in a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                    "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
]

# Prompts kept in German as measured; translating them changes token counts.
INTERAKTIV = [
    ("Erkläre in drei Sätzen, was ein Mixture-of-Experts-Modell ist.", 150),
    ("Schreib eine Python-Funktion, die prüft, ob eine Zeichenkette ein Palindrom ist, mit kurzem Docstring.", 250),
    ("Was ist der Unterschied zwischen TCP und UDP? Kurz.", 150),
    ("Gib mir ein zsh-Einzeiler, der alle .log-Dateien älter als 7 Tage findet und löscht, und erkläre ihn.", 200),
    ("Fasse die Vor- und Nachteile von SQLite gegenüber PostgreSQL für eine kleine Web-App zusammen.", 300),
]

# Prompts kept in German as measured; translating them changes token counts.
# Agent turns: one task, then every turn returns a real file as the tool result (growing history).
AGENT_TASK = ("Im Repository llama.cpp soll der HTTP-Server einen neuen Query-Parameter bekommen. "
              "Finde heraus, wo Requests geparst werden, und schlage eine minimale Änderung vor. "
              "Benutze die Tools, eine Datei pro Schritt.")
AGENT_FILES = ["tools/server/server-common.cpp", "tools/server/server-http.cpp", "common/arg.h",
               "tools/server/server-common.h", "common/common.h", "tools/server/server-task.h"]


def tool_result_text(path, limit_tok=1500):
    p = os.path.join(SRC, path)
    return open(p, errors="replace").read()[: int(limit_tok * 3.2)]  # a missing file aborts instead of silently producing a shorter history


def run_interaktiv(log, args):
    out = []
    sysmsg = "Du bist ein hilfreicher Assistent. Antworte knapp."
    if args.einmalig: sysmsg = f"[{uuid.uuid4().hex}] {sysmsg}"  # prefix unique per pass
    for q, mt in INTERAKTIV:
        r = chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": q}], mt, thinking=args.thinking)
        r.update(scenario="interaktiv", q=q[:40]); log(r); out.append(r)
    return out


def run_agent(log, args, ctx_tokens=None, cold=True):
    ctx_tokens = ctx_tokens or args.agent_ctx
    nonce = args.nonce or (uuid.uuid4().hex if cold else "fest")
    base = corpus(ctx_tokens, "tools/server/*.cpp")
    sysmsg = (f"[{nonce}] Du bist ein Coding-Agent im Repository llama.cpp. Arbeite schrittweise mit den Tools. "
              "Hier ist der relevante Quelltext als Kontext:\n\n")
    sysmsg += trim_to(base, ctx_tokens)
    msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": AGENT_TASK}]
    out = []; t_all = time.time()
    for i, path in enumerate(AGENT_FILES):
        r = chat(msgs, args.agent_max_tokens, tools=TOOLS, thinking=args.thinking)
        r.update(scenario="agent", turn=i + 1, ctx_target=ctx_tokens); log(r); out.append(r)
        if r["error"]: break
        if r["tool_calls"]:
            tc = r["tool_calls"][0]; tcid = tc["id"] or f"call_{i}"
            msgs.append({"role": "assistant", "content": r["text"] or "",
                         "tool_calls": [{"id": tcid, "type": "function",
                                         "function": {"name": tc["name"], "arguments": tc["arguments"]}}]})
            msgs.append({"role": "tool", "tool_call_id": tcid, "content": tool_result_text(path)})
        else:  # no tool call: keep the answer, supply the file as the next user message
            msgs.append({"role": "assistant", "content": r["text"]})
            msgs.append({"role": "user", "content": f"Inhalt von {path}:\n" + tool_result_text(path)})
    total = time.time() - t_all
    log({"scenario": "agent", "summary": True, "total_s": total, "turns": len(out)})
    return out, total


def run_dokument(log, args):
    out = []
    doc = corpus(max(args.doc_ctx) + 2000, "**/*.md")
    for n in args.doc_ctx:  # nonce per length: the shorter document is a prefix of the longer one, otherwise the longer one would not be cold
        text = trim_to(doc, n)
        msgs = [{"role": "system", "content": f"[{args.nonce + '-' + str(n) if args.nonce else uuid.uuid4().hex}] Du beantwortest Fragen zu einer Dokumentation."},
                {"role": "user", "content": text + "\n\nFrage: Welche Kommandozeilenoptionen des Servers werden "
                                                   "in diesem Dokument am häufigsten erwähnt? Nenne fünf."}]
        r = chat(msgs, 200, thinking=args.thinking, vram=True)
        r.update(scenario="dokument", ctx_target=n); log(r); out.append(r)
    return out


def run_repetitiv(log, args):
    """The answer repeats the prompt almost verbatim: best case for ngram-mod. Prompts are identical in all runs."""
    if args.art == "json":
        rnd = random.Random(6003)
        daten = [{"id": i, "name": f"dienst-{rnd.randrange(10**6):06d}", "port": rnd.randrange(1024, 65535), "aktiv": True,
                  "tags": rnd.sample(["http", "grpc", "intern", "extern", "batch", "stream", "cache", "db"], 3)} for i in range(60)]
        text = "Gib dieses JSON vollständig und unverändert zurück, nur bei jedem Objekt steht aktiv auf false. Keine Erklärung.\n\n" + json.dumps(daten, ensure_ascii=False, indent=1)
    else:
        src = open(os.path.join(SRC, "common/ngram-mod.cpp"), errors="replace").read()
        text = "Gib diesen Quelltext vollständig zurück und benenne nur die Variable used in n_used um. Keine Erklärung.\n\n" + src
    out = []
    for run in range(args.runs):  # same prompt every run, so from run 2 on only tg matters: prefill does not disturb the comparison
        r = chat([{"role": "system", "content": "Du bearbeitest Dateien exakt nach Anweisung."}, {"role": "user", "content": text}], args.spec_tokens)
        r.update(scenario="repetitiv", art=args.art, lauf=run + 1); log(r); out.append(r)
    return out


# Prompts kept in German as measured; translating them changes token counts.
NEUARTIG = ["Schreib eine kurze, ungewöhnliche Geschichte über eine Leuchtturmwärterin, die Schachzüge per Morsezeichen tauscht.",
            "Erkläre einem Kind ohne Fachwörter, warum der Himmel bei Sonnenuntergang rot wird, in eigenen Vergleichen.",
            "Entwirf drei unterschiedliche Argumente für und gegen eine Stadt ohne Parkplätze und wäge sie ab."]


def run_neuartig(log, args):
    out = []
    for q in NEUARTIG:
        r = chat([{"role": "system", "content": "Du schreibst frei und wiederholst dich nicht."}, {"role": "user", "content": q}], 400)
        r.update(scenario="neuartig", q=q[:40]); log(r); out.append(r)
    return out


def summary(rows, total_agent):
    def med(xs):
        xs = [x for x in xs if x is not None]; return statistics.median(xs) if xs else float("nan")
    it = [r for r in rows if r.get("scenario") == "interaktiv"]
    if it:
        print(f"interaktiv  n={len(it)}  TTFT median {med(r['ttft'] for r in it):.2f} s  "
              f"total median {med(r['wall'] for r in it):.2f} s  tg median {med(r['tg_tps'] for r in it):.1f} t/s  "
              f"errors {sum(1 for r in it if r['error'])}")
    ag = [r for r in rows if r.get("scenario") == "agent" and not r.get("summary")]
    if ag:
        print("agent       turn  wall_s  ttft_s  prompt_n  cache_n  pp_t/s  gen  tg_t/s")
        for r in ag:
            print(f"            {r['turn']:4d}  {r['wall']:6.1f}  {r['ttft'] or 0:6.2f}  {r['prompt_n'] or 0:8d}  "
                  f"{r['cache_n'] or 0:7d}  {r['pp_tps'] or 0:6.0f}  {r['predicted_n'] or 0:3d}  {r['tg_tps'] or 0:6.1f}"
                  f"{'  ' + r['error'] if r['error'] else ''}")
        for k, v in total_agent: print(f"            total ({k}): {v:.1f} s")
    do = [r for r in rows if r.get("scenario") == "dokument"]
    for r in do:
        print(f"dokument    ctx {r['ctx_target']:6d}  prompt_n {r['prompt_n']}  prefill {(r['prompt_ms'] or 0)/1000:.1f} s "
              f"({r['pp_tps'] or 0:.0f} t/s)  wall {r['wall']:.1f} s  VRAM peak {r.get('vram_peak_mib')} MiB  "
              f"{r['error'] or 'ok'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("szenario", nargs="?", default="alle", choices=["interaktiv", "agent", "dokument", "alle", "repetitiv", "neuartig"])
    ap.add_argument("--art", default="json", choices=["json", "code"]); ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--spec-tokens", type=int, default=1500)
    ap.add_argument("--out", default=None); ap.add_argument("--label", default="replay")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--einmalig", action="store_true", help="interaktiv: use a separate nonce in the system prompt for every pass")
    ap.add_argument("--nonce", default=None, help="fixed nonce: identical prompts across processes, prefix stays cacheable")
    ap.add_argument("--agent-ctx", type=int, default=15000)
    ap.add_argument("--agent-max-tokens", type=int, default=300)
    ap.add_argument("--doc-ctx", type=lambda s: [int(x) for x in s.split(",")], default=[32000, 64000])
    ap.add_argument("--loop", type=float, default=0, help="sustained load: repeat the scenarios for this many seconds")
    args = ap.parse_args()
    props = json.load(http("/props"))
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench", args.label, f"replay-{time.strftime('%H%M%S')}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fh = open(out, "a")
    def log(r):
        r = dict(r, ts=time.time(), label=args.label, n_ctx=n_ctx)
        if "text" in r: r["text_len"] = len(r["text"]); r["text"] = r["text"][:200]
        fh.write(json.dumps(r, ensure_ascii=False) + "\n"); fh.flush()
    print(f"# {args.label}  n_ctx={n_ctx}  → {out}", flush=True)
    t_end = time.time() + args.loop; rows = []; totals = []; it = 0
    while True:
        it += 1
        if args.szenario in ("interaktiv", "alle"): rows += run_interaktiv(log, args)
        if args.szenario in ("agent", "alle"):
            o, t = run_agent(log, args, cold=True); rows += o; totals.append(("kalt", t))  # "kalt" = cold
        if args.szenario in ("dokument", "alle"):
            rows += run_dokument(log, args)
        if args.szenario == "repetitiv": rows += run_repetitiv(log, args)
        if args.szenario == "neuartig": rows += run_neuartig(log, args)
        if not args.loop or time.time() > t_end: break
    summary(rows, totals)
    errs = sum(1 for r in rows if r.get("error"))
    print(f"# {len(rows)} requests in {it} passes, {errs} errors")


if __name__ == "__main__":
    main()
