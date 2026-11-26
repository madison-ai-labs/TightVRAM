#!/usr/bin/env python3
"""Build the trace prompts with the model's embedded chat template (jinja2, gguf-py from a llama.cpp checkout).

Input:  env LLAMA_SRC (llama.cpp checkout, default ~/src/llama.cpp), env MODEL (GGUF with chat template,
        default ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf), env DOCS_DIR (Markdown files for prompt 4,
        default $LLAMA_SRC/docs), env REVIEW_FILE (source file for prompt 6,
        default $LLAMA_SRC/tools/server/server-context.cpp), agent-base.json (agent request: tools + task).
Output: prompts/<name>.txt (fully rendered, with special tokens) and prompts/meta.json
        (n_predict per prompt; "zeichen" = character count of the rendered prompt).
Run:    python3 make-prompts.py     (needs: pip install jinja2)
Contents: typical workloads: an agent request with tools, llama.cpp source, code writing, a German doc question,
short English chat, code review.
"""
import glob, json, os, sys, datetime
SRC = os.path.expanduser(os.environ.get("LLAMA_SRC", "~/src/llama.cpp"))
sys.path.insert(0, os.path.join(SRC, "gguf-py"))
import gguf, jinja2

MODEL = os.path.expanduser(os.environ.get("MODEL", "~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"))
D = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "prompts")
DOCS_DIR = os.path.expanduser(os.environ.get("DOCS_DIR", os.path.join(SRC, "docs")))
REVIEW_FILE = os.path.expanduser(os.environ.get("REVIEW_FILE", os.path.join(SRC, "tools/server/server-context.cpp")))

r = gguf.GGUFReader(MODEL)
f = r.fields["tokenizer.chat_template"]
tmpl_src = bytes(f.parts[f.data[0]]).decode()

def raise_exception(msg): raise RuntimeError(msg)
env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, extensions=["jinja2.ext.loopcontrols"])
env.globals["raise_exception"] = raise_exception
env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
env.filters["tojson"] = lambda x, indent=None, ensure_ascii=False: json.dumps(x, indent=indent, ensure_ascii=ensure_ascii)
tmpl = env.from_string(tmpl_src)

def render(messages, tools=None, thinking=False):
    return tmpl.render(messages=messages, tools=tools, add_generation_prompt=True, enable_thinking=thinking,
                       bos_token="", eos_token="<|im_end|>")

def read(p, limit=None):
    s = open(p, errors="replace").read()
    return s[:limit] if limit else s

base = json.load(open(os.path.join(D, "agent-base.json")))
tools = base["tools"]
prompts = {}

# NOTE: the German prompt texts below are the measured workload and must stay unchanged.
# 1. Agent request as in daily use (tools), thinking as in the original request
prompts["agent-tools"] = (render(base["messages"], tools, base.get("chat_template_kwargs", {}).get("enable_thinking", True)), 700)

# 2. Agent with a large source context: read build_moe_ffn and plan a change
g = read(f"{SRC}/src/llama-graph.cpp")
i = g.find("ggml_tensor * llm_graph_context::build_moe_ffn")
ctx_src = g[max(0, i - 20000): i + 25000]
prompts["agent-quelltext"] = (render([
    {"role": "system", "content": base["messages"][0]["content"]},
    {"role": "user", "content": "Hier ist ein Ausschnitt aus src/llama-graph.cpp:\n\n```cpp\n" + ctx_src +
     "\n```\n\nIch will die Experten eines Layers auf zwei Tensoren aufteilen: häufige auf der GPU, seltene auf der CPU. "
     "Welche Stellen in build_moe_ffn müssen sich ändern? Skizziere den Patch."}], tools, True), 900)

# 3. Write code without context
prompts["code-neu"] = (render([{"role": "user", "content":
    "Schreibe ein Python-Modul expert_cache.py: ein Cache für MoE-Experten pro Layer mit kombinierter LRU/LFU-Verdrängung, "
    "fest gepinnten Einträgen und Trefferstatistik. Dazu pytest-Tests. Nur Code, kurze Kommentare."}], None, False), 900)

# 4. Doc question over Markdown files from DOCS_DIR. The question stays in German on purpose: the point of this
# prompt is German-language routing (German tokens may route to different experts than English ones).
docs = sorted(glob.glob(os.path.join(DOCS_DIR, "*.md")))
doc = "\n\n".join(f"### {os.path.basename(p)}\n{read(p, 14000)}" for p in docs)[:40000]
prompts["doku-deutsch"] = (render([
    {"role": "system", "content": "Du beantwortest Fragen nur aus den gegebenen Dokumenten und nennst die Quelle."},
    {"role": "user", "content": doc + "\n\nFasse die drei wichtigsten Betriebsentscheidungen zusammen und sag, welche davon sich widersprechen könnten."}],
    None, False), 600)

# 5. Short chat, English, with thinking
prompts["chat-en"] = (render([{"role": "user", "content":
    "Why does speculative decoding help less for mixture-of-experts models on GPUs, and when could it help more? Be concrete."}],
    None, True), 600)

# 6. Code review of a foreign source file (REVIEW_FILE)
rev = read(REVIEW_FILE, 30000)
prompts["review"] = (render([{"role": "user", "content":
    "Review diesen Code auf Fehler und riskante Stellen, priorisiert:\n\n```\n" + rev + "\n```"}], None, False), 700)

os.makedirs(OUT, exist_ok=True)
meta = {}
for name, (text, n) in prompts.items():
    open(f"{OUT}/{name}.txt", "w").write(text)
    meta[name] = {"n_predict": n, "zeichen": len(text)}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=1)
print(json.dumps(meta, indent=1))
