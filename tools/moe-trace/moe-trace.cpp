// moe-trace: records MoE routing per layer and token (expert IDs and router weights).
// Prefills the prompt from -f, then generates -n tokens greedily. Based on examples/eval-callback.
// Input:  usual llama.cpp common args (-m, -f, -n, -t, ...), env MOE_TRACE_OUT = output file.
// Output: MOE_TRACE_OUT, binary, little endian, a sequence of records with an int32 header [kind, a, b, c]:
//   kind 2: marker before each llama_process: a = phase (0 prefill, 1 decode), b = n_tokens, c = pos0
//   kind 0: ffn_moe_topk  layer a, ne0 = b (n_expert_used), ne1 = c (n_tokens), followed by b*c int32
//   kind 1: ffn_moe_weights* layer a, ne0 = b, ne1 = c, followed by b*c float32 (last version per layer counts)
//   kind 3: generated token, a = token id
// Build: see build.sh. Run: see run.sh; read the output with analyze.py. CPU only (-ngl 0) next to a running
// server; routing does not depend on the backend (apart from numeric noise at close top-k boundaries).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static FILE * g_out = nullptr;

static void put4(int32_t k, int32_t a, int32_t b, int32_t c) {
    int32_t h[4] = {k, a, b, c};
    fwrite(h, sizeof(h), 1, g_out);
}

static int layer_of(const char * name, const char * prefix) {
    size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || name[n] != '-') return -1;
    return atoi(name + n + 1);
}

static bool cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    const char * name = ggml_get_name(t);
    int il_k = layer_of(name, "ffn_moe_topk");
    int il_w = -1;
    // Weights after normalization if present, else raw. The parser takes the last one per layer and ubatch.
    for (const char * p : {"ffn_moe_weights", "ffn_moe_weights_norm", "ffn_moe_weights_softmax", "ffn_moe_weights_scaled"}) {
        int il = layer_of(name, p);
        if (il >= 0) { il_w = il; break; }
    }
    if (ask) return il_k >= 0 || il_w >= 0;
    if (il_k < 0 && il_w < 0) return true;

    const int64_t ne0 = t->ne[0], ne1 = t->ne[1] * t->ne[2] * t->ne[3];
    const bool is_k = il_k >= 0;
    if (is_k && t->type != GGML_TYPE_I32) { LOG_ERR("topk type %d\n", t->type); return true; }
    if (!is_k && t->type != GGML_TYPE_F32) { LOG_ERR("weights type %d\n", t->type); return true; }
    // ffn_moe_topk is a view on the argsort (nb1 = n_expert * 4), so it is not contiguous:
    // gather row by row along the strides. Within a row it must be dense.
    const size_t es = ggml_type_size(t->type);
    if (t->nb[0] != es) { LOG_ERR("%s nb0 %zu != %zu\n", name, t->nb[0], es); return true; }
    const size_t row = (size_t) ne0 * es;
    std::vector<uint8_t> buf((size_t) ne0 * ne1 * es);
    const bool host = !t->buffer || ggml_backend_buffer_is_host(t->buffer);
    size_t o = 0;
    for (int64_t i3 = 0; i3 < t->ne[3]; i3++)
    for (int64_t i2 = 0; i2 < t->ne[2]; i2++)
    for (int64_t i1 = 0; i1 < t->ne[1]; i1++, o += row) {
        const size_t off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        if (host) memcpy(buf.data() + o, (const uint8_t *) t->data + off, row);
        else      ggml_backend_tensor_get(t, buf.data() + o, off, row);
    }
    // ffn_moe_weights has shape [1, n_expert_used, n_tokens]: ne0 = 1 -> remap
    int64_t e0 = ne0, e1 = ne1;
    if (!is_k && ne0 == 1) { e0 = t->ne[1]; e1 = t->ne[2] * t->ne[3]; }
    put4(is_k ? 0 : 1, is_k ? il_k : il_w, (int32_t) e0, (int32_t) e1);
    fwrite(buf.data(), 1, buf.size(), g_out);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) return 1;

    const char * out = getenv("MOE_TRACE_OUT");
    if (!out) { fprintf(stderr, "MOE_TRACE_OUT missing\n"); return 1; }
    g_out = fopen(out, "wb");
    if (!g_out) { perror(out); return 1; }

    llama_backend_init();
    llama_numa_init(params.numa);
    params.cb_eval = cb;
    params.cb_eval_user_data = nullptr;
    params.warmup = false;

    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    if (!model || !ctx) { LOG_ERR("init failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (toks.empty()) { LOG_ERR("empty prompt\n"); return 1; }
    const int n_batch = params.n_batch > 0 ? params.n_batch : 2048;
    fprintf(stderr, "moe-trace: %zu prompt tokens, n_batch %d, n_predict %d\n", toks.size(), n_batch, params.n_predict);

    int pos = 0;
    for (size_t off = 0; off < toks.size(); off += n_batch) {
        int n = (int) std::min<size_t>(n_batch, toks.size() - off);
        common_batch b = common_batch_get_one(ctx, toks.data() + off, n);
        put4(2, 0, n, pos);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get())) { LOG_ERR("prefill failed at %d\n", pos); return 1; }
        pos += n;
    }

    int n_gen = 0;
    for (; n_gen < params.n_predict; ++n_gen) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
        put4(3, best, 0, 0);
        if (llama_vocab_is_eog(vocab, best)) break;
        llama_token tk = best;
        common_batch b = common_batch_get_one(ctx, &tk, 1);
        put4(2, 1, 1, pos);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, b.get())) { LOG_ERR("decode failed at %d\n", pos); return 1; }
        pos += 1;
    }
    fclose(g_out);
    fprintf(stderr, "moe-trace: done, %d tokens generated\n", n_gen);
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
