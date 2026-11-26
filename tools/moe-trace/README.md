# moe-trace

Records which experts a MoE model routes to, per layer and token, and simulates expert-caching strategies on those traces.
Targets Qwen3.6-35B-A3B (40 layers, 256 experts, top-8); the analysis constants live at the top of `analyze.py`.

Environment variables (all optional):

| Variable | Default | Used by |
|---|---|---|
| `LLAMA_SRC` | `~/src/llama.cpp` | build.sh, make-prompts.py |
| `MODEL` | `~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | make-prompts.py, run.sh |
| `DOCS_DIR` | `$LLAMA_SRC/docs` | make-prompts.py (prompt 4, `*.md`) |
| `REVIEW_FILE` | `$LLAMA_SRC/tools/server/server-context.cpp` | make-prompts.py (prompt 6) |
| `T`, `PROMPTS`, `DEADLINE` | 6 threads, all prompts, none | run.sh |

## Steps

1. **Build** against an existing llama.cpp build (`$LLAMA_SRC/build`): `./build.sh` produces `./moe-trace`.
2. **Make prompts**: `python3 make-prompts.py` (needs `jinja2`) renders six prompts with the model's chat template into
   `prompts/*.txt` plus `prompts/meta.json`.
3. **Run**: `./run.sh` traces every prompt on the CPU (`-ngl 0`) and writes `traces/<prompt>.bin`. Timings from this run
   are meaningless; only the routing is used.
4. **Analyze**: `python3 analyze.py traces/ [--json out.json]` prints hit rates of a fixed hot-expert set
   (leave-one-prompt-out), decode reuse, batch unions and router-weight statistics.
5. **Fork simulation**: `python3 fork-sim.py traces/ [--json out.json]` replays an adaptive expert-swap strategy
   (decayed scores, one swap per layer per update, hysteresis) and estimates tg.

## Notes

- Prompt 4 (doc question) is intentionally German: it tests German-language routing. Its documents come from `DOCS_DIR`,
  so the trace differs from the original private-docs run. Prompt 6 uses `REVIEW_FILE` likewise.
- `agent-base.json`: the original system prompt was replaced by a neutral English one, and private names in the
  tool descriptions were neutralized. Prompts `agent-tools` and `agent-quelltext` therefore do not reproduce the
  original traces byte for byte. The German user message and tool descriptions were kept as they were.
- Trace format: see the header of `moe-trace.cpp`. `meta.json` keys (`n_predict`, `zeichen`) are kept from the
  original German version.
