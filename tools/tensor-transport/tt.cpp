// tt - tensor-transport benchmark (RTX 3070 8 GB, PCIe 3.0, DDR4, NVMe).
//
// Measures real expert tensors of Qwen3.6-35B-A3B per quantization format.
// Subcommands (all print tab-separated rows on stdout, one row per measurement):
//   quant  quality vs. BF16 (relative Frobenius error) and bytes, also for base+residual,
//          outlier channel and shared base + delta (D^2-MoE style). Writes the quantized tensors.
//   cpu    MUL_MAT_ID on the CPU (tg path: experts live in RAM), random experts per repetition.
//   gpu    MUL_MAT_ID on the GPU, weights in VRAM, n tokens (1 = decode, 4096 = prefill ubatch).
//   h2d    upload of a whole tensor host->VRAM, pinned and pageable.
//   ssd    O_DIRECT reads from the NVMe (format independent; GB/s per block size and thread count).
//
// Composite formats are written "A+B[+C]": sum of several MUL_MAT_ID of the same shape (base + residuals).
//
// Build: see build.sh (needs a llama.cpp checkout built with CUDA; set LLAMA_SRC).
// Run:   see run-quant.sh and run-timing.sh. Output column names (art, tensor, quant_modus, ...) are German
//        or mixed and are kept as-is because existing data files and analyze.py depend on them.
//        Values: "ausgeliefert:<type>" = reference quantization shipped in the comparison GGUF,
//        "gepinnt" = pinned host memory, "ohne" = no imatrix, "sek" = seconds, "gelesen" = bytes read.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <fcntl.h>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using clk = std::chrono::steady_clock;
static double sek(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

static void die(const char * m) { fprintf(stderr, "%s\n", m); exit(1); }

static ggml_type typ_aus_name(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; t++) {
        const auto * tr = ggml_get_type_traits((ggml_type) t);
        if (tr->type_name && strcasecmp(s.c_str(), tr->type_name) == 0) return (ggml_type) t;
    }
    die(("unknown type " + s).c_str());
    return GGML_TYPE_F32;
}

static std::vector<std::string> teile(const std::string & s, char c) {
    std::vector<std::string> r; size_t a = 0;
    for (;;) { size_t b = s.find(c, a); r.push_back(s.substr(a, b - a)); if (b == std::string::npos) break; a = b + 1; }
    return r;
}

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

// ---------- tensor files: header (type, ne0, ne1, ne2) + raw data ----------

struct Datei { ggml_type typ; int64_t ne[3]; std::vector<uint8_t> d; };

static void schreibe(const std::string & pfad, ggml_type t, const int64_t * ne, const void * d, size_t n) {
    FILE * f = fopen(pfad.c_str(), "wb"); if (!f) die(pfad.c_str());
    int64_t k[4] = { (int64_t) t, ne[0], ne[1], ne[2] };
    fwrite(k, sizeof k, 1, f); fwrite(d, 1, n, f); fclose(f);
}

static Datei lade(const std::string & pfad) {
    FILE * f = fopen(pfad.c_str(), "rb"); if (!f) die(("missing: " + pfad).c_str());
    int64_t k[4]; if (fread(k, sizeof k, 1, f) != 1) die("header");
    Datei r; r.typ = (ggml_type) k[0]; r.ne[0] = k[1]; r.ne[1] = k[2]; r.ne[2] = k[3];
    size_t n = ggml_row_size(r.typ, r.ne[0]) * r.ne[1] * r.ne[2];
    r.d.resize(n); if (fread(r.d.data(), 1, n, f) != n) die("data"); fclose(f);
    return r;
}

// ---------- quant ----------

static void parallel(int64_t n, int nth, const std::function<void(int64_t, int64_t)> & f) {
    std::vector<std::thread> th; int64_t step = (n + nth - 1) / nth;
    for (int i = 0; i < nth; i++) { int64_t a = i * step, b = std::min(n, a + step); if (a < b) th.emplace_back(f, a, b); }
    for (auto & t : th) t.join();
}

// imatrix (unsloth): per expert, sum of x^2 per input column (in_sum2) and token count (counts).
// g_s2 = raw in_sum2 [E][k] for the weighted error, g_im = in_sum2/counts for quantization.
static std::vector<float> g_s2, g_im;
static bool g_im_quant = false;

// Quantizes src (nrows x k) to type t and writes the dequantized result to out. Returns bytes.
// rpe = rows per expert (to pick the imatrix row), 0 = no imatrix.
static size_t quant_deq(ggml_type t, const float * src, float * out, int64_t nrows, int64_t k, std::vector<uint8_t> * roh, int nth, int64_t rpe = 0) {
    size_t rs = ggml_row_size(t, k);
    std::vector<uint8_t> q(rs * nrows);
    bool im = g_im_quant && rpe > 0 && !g_im.empty();
    if (ggml_quantize_requires_imatrix(t) && !im) die("type needs an imatrix");
    if (im) {
        int64_t ne_ = nrows / rpe;
        parallel(ne_, nth, [&](int64_t a, int64_t b) {
            for (int64_t e = a; e < b; e++)
                ggml_quantize_chunk(t, src + e * rpe * k, q.data() + e * rpe * rs, 0, rpe, k, g_im.data() + e * k);
        });
    } else {
        // row-wise chunks are always valid for ggml_quantize_chunk
        parallel(nrows, nth, [&](int64_t a, int64_t b) {
            ggml_quantize_chunk(t, src + a * k, q.data() + a * rs, 0, b - a, k, nullptr);
        });
    }
    const auto * tr = ggml_get_type_traits(t);
    parallel(nrows, nth, [&](int64_t a, int64_t b) {
        for (int64_t r = a; r < b; r++) tr->to_float(q.data() + r * rs, out + r * k, k);
    });
    if (roh) roh->swap(q);
    return rs * nrows;
}

// Error weighted by sum x^2 per column and expert: ~ E|dW.x|^2 / E|W.x|^2 over the calibration tokens (diagonal approximation).
static double relfehler_akt(const float * w, const float * wq, int64_t k, int64_t rpe, int64_t E) {
    if (g_s2.empty()) return NAN;
    std::vector<double> e_(E), s_(E);
    parallel(E, 10, [&](int64_t a, int64_t b) {
        for (int64_t e = a; e < b; e++) {
            double ee = 0, ss = 0; const float * s2 = g_s2.data() + e * k;
            for (int64_t r = 0; r < rpe; r++) {
                const float * wr = w + (e * rpe + r) * k; const float * qr = wq + (e * rpe + r) * k;
                for (int64_t j = 0; j < k; j++) { double d = (double) wr[j] - qr[j]; ee += s2[j] * d * d; ss += s2[j] * (double) wr[j] * wr[j]; }
            }
            e_[e] = ee; s_[e] = ss;
        }
    });
    double ee = 0, ss = 0; for (int64_t e = 0; e < E; e++) { ee += e_[e]; ss += s_[e]; }
    return std::sqrt(ee / ss);
}

static double relfehler(const float * w, const float * wq, size_t n) {
    double e = 0, s = 0;
    for (size_t i = 0; i < n; i++) { double d = (double) w[i] - wq[i]; e += d * d; s += (double) w[i] * w[i]; }
    return std::sqrt(e / s);
}

static std::vector<float> lade_gguf_f32(const char * pfad, const char * name, int64_t * ne, ggml_type * typ_out) {
    gguf_init_params p = { true, nullptr };
    gguf_context * g = gguf_init_from_file(pfad, p); if (!g) die("gguf");
    int64_t id = gguf_find_tensor(g, name); if (id < 0) die(("tensor missing: " + std::string(name)).c_str());
    ggml_type t = gguf_get_tensor_type(g, id);
    size_t off = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id), n = gguf_get_tensor_size(g, id);
    // shape from a metadata-only context
    ggml_context * mc = nullptr; gguf_init_params p2 = { true, &mc };
    gguf_context * g2 = gguf_init_from_file(pfad, p2);
    ggml_tensor * tt = ggml_get_tensor(mc, name);
    for (int i = 0; i < 3; i++) ne[i] = tt->ne[i];
    std::vector<uint8_t> roh(n);
    int fd = open(pfad, O_RDONLY); size_t got = 0;
    while (got < n) { ssize_t r = pread(fd, roh.data() + got, n - got, off + got); if (r <= 0) die("pread"); got += r; }
    close(fd);
    int64_t k = ne[0], rows = ne[1] * ne[2];
    std::vector<float> f(k * rows);
    const auto * tr = ggml_get_type_traits(t);
    size_t rs = ggml_row_size(t, k);
    if (t == GGML_TYPE_F32) memcpy(f.data(), roh.data(), n);
    else parallel(rows, 10, [&](int64_t a, int64_t b) { for (int64_t r = a; r < b; r++) tr->to_float(roh.data() + r * rs, f.data() + r * k, k); });
    if (typ_out) *typ_out = t;
    gguf_free(g); gguf_free(g2); ggml_free(mc);
    return f;
}

// quant REF.gguf NAME OUTDIR [COMPARE.gguf] -- SPEC...
// SPEC: TYPE | TYPE+TYPE[+TYPE] (residuals) | TYPE+outP (outliers, P in per mille) | shared:TYPE+TYPE (mean base + delta)
static int cmd_quant(int argc, char ** argv) {
    if (argc < 5) die("quant REF NAME OUTDIR [COMPARE] -- SPEC...");
    const char * ref = argv[1]; const char * name = argv[2]; std::string out = argv[3];
    int i = 4; const char * vgl = nullptr;
    if (strcmp(argv[i], "--") != 0) vgl = argv[i++];
    i++;
    int nth = 10;
    int64_t ne[3]; ggml_type rt;
    auto t0 = clk::now();
    std::vector<float> w = lade_gguf_f32(ref, name, ne, &rt);
    int64_t k = ne[0], rows = ne[1] * ne[2]; size_t n = (size_t) k * rows;
    fprintf(stderr, "%s %s [%ld,%ld,%ld] loaded in %.1f s\n", name, ggml_type_name(rt), ne[0], ne[1], ne[2], sek(t0, clk::now()));
    double nparam = (double) n;
    int64_t rpe = ne[1], E = ne[2];
    if (const char * ip = getenv("TT_IMATRIX")) {
        int64_t ie[3], ce[3];
        g_s2 = lade_gguf_f32(ip, (std::string(name) + ".in_sum2").c_str(), ie, nullptr);
        std::vector<float> cnt = lade_gguf_f32(ip, (std::string(name) + ".counts").c_str(), ce, nullptr);
        if (ie[0] != k || ie[1] != E) die("imatrix shape does not match");
        g_im.resize(g_s2.size());
        int64_t leer = 0;
        for (int64_t e = 0; e < E; e++) for (int64_t j = 0; j < k; j++) {
            float c = cnt[e]; if (c <= 0 && j == 0) leer++;
            g_im[e * k + j] = c > 0 ? g_s2[e * k + j] / c : 1.0f;
        }
        g_im_quant = !getenv("TT_IMATRIX_QUANT") || atoi(getenv("TT_IMATRIX_QUANT")) != 0;
        fprintf(stderr, "imatrix %s: %ld experts without tokens, quantizing with imatrix: %d\n", ip, leer, g_im_quant);
    }
    const char * modus = g_im_quant ? "imatrix" : "ohne";
    printf("art\ttensor\tquant_modus\tspec\tbytes\tbpw\trelfehler\tsqnr_db\trelfehler_akt\tsqnr_akt_db\n");
    auto zeile = [&](const std::string & spec, size_t bytes, const float * wq) {
        double rel = relfehler(w.data(), wq, n), ra = relfehler_akt(w.data(), wq, k, rpe, E);
        printf("quant\t%s\t%s\t%s\t%zu\t%.3f\t%.5f\t%.2f\t%.5f\t%.2f\n", name, modus, spec.c_str(), bytes, bytes * 8.0 / nparam,
               rel, -20 * std::log10(rel), ra, -20 * std::log10(ra));
        fflush(stdout);
    };
    if (vgl) {  // shipped quantization (with imatrix) as reference
        int64_t ne2[3]; ggml_type vt;
        std::vector<float> v = lade_gguf_f32(vgl, name, ne2, &vt);
        zeile(std::string("ausgeliefert:") + ggml_type_name(vt), ggml_row_size(vt, k) * rows, v.data());
    }
    std::vector<float> deq(n), rest(n), summe(n);
    for (; i < argc; i++) {
        std::string spec = argv[i];
        t0 = clk::now();
        bool shared = spec.rfind("shared:", 0) == 0;
        auto teile_ = teile(shared ? spec.substr(7) : spec, '+');
        size_t bytes = 0;
        std::fill(summe.begin(), summe.end(), 0.0f);
        if (shared) {
            // base = mean over experts (one tensor [k, ne1]), delta per expert
            int64_t per = k * ne[1]; std::vector<float> mittel(per, 0.0f), md(per);
            for (int64_t e = 0; e < ne[2]; e++) for (int64_t j = 0; j < per; j++) mittel[j] += w[e * per + j];
            for (auto & x : mittel) x /= ne[2];
            double dn = 0, wn = 0;
            for (int64_t e = 0; e < ne[2]; e++) for (int64_t j = 0; j < per; j++) { double d = w[e * per + j] - mittel[j]; dn += d * d; wn += (double) w[e * per + j] * w[e * per + j]; }
            printf("info\t%s\tdelta_anteil\t%.4f\n", name, std::sqrt(dn / wn));
            bytes += quant_deq(typ_aus_name(teile_[0]), mittel.data(), md.data(), ne[1], k, nullptr, nth);
            for (int64_t e = 0; e < ne[2]; e++) for (int64_t j = 0; j < per; j++) { summe[e * per + j] = md[j]; rest[e * per + j] = w[e * per + j] - md[j]; }
            for (size_t p = 1; p < teile_.size(); p++) {
                bytes += quant_deq(typ_aus_name(teile_[p]), rest.data(), deq.data(), rows, k, nullptr, nth, rpe);
                for (size_t j = 0; j < n; j++) { summe[j] += deq[j]; rest[j] -= deq[j]; }
            }
        } else {
            memcpy(rest.data(), w.data(), n * sizeof(float));
            for (size_t p = 0; p < teile_.size(); p++) {
                const std::string & tp = teile_[p];
                if (tp.rfind("out", 0) == 0) {
                    // outliers: the largest |rest| (per mille) kept exactly, 6 bytes per entry (fp16 + u32 index)
                    double prom = atof(tp.c_str() + 3);
                    size_t m = (size_t) (n * prom / 1000.0);
                    std::vector<float> a(n); for (size_t j = 0; j < n; j++) a[j] = std::fabs(rest[j]);
                    std::nth_element(a.begin(), a.begin() + (n - m), a.end());
                    float schwelle = a[n - m];
                    for (size_t j = 0; j < n; j++) if (std::fabs(rest[j]) >= schwelle) { summe[j] += rest[j]; rest[j] = 0; }
                    bytes += m * 6;
                    continue;
                }
                std::vector<uint8_t> roh;
                ggml_type t = typ_aus_name(tp);
                bytes += quant_deq(t, rest.data(), deq.data(), rows, k, teile_.size() == 1 ? &roh : nullptr, nth, rpe);
                for (size_t j = 0; j < n; j++) { summe[j] += deq[j]; rest[j] -= deq[j]; }
                if (teile_.size() == 1) schreibe(out + "/" + name + "." + tp + ".bin", t, ne, roh.data(), roh.size());
            }
        }
        zeile(spec, bytes, summe.data());
        fprintf(stderr, "  %s %.1f s\n", spec.c_str(), sek(t0, clk::now()));
    }
    return 0;
}

// ---------- compute: MUL_MAT_ID on CPU or GPU ----------

struct Lauf {
    ggml_backend_t be; ggml_context * ctx; ggml_backend_buffer_t buf; ggml_gallocr_t ga; ggml_cgraph * gf;
    ggml_tensor * ids; ggml_tensor * x; int64_t ntok, nused, E; size_t wbytes;
};

static Lauf baue(ggml_backend_t be, const std::string & dir, const std::string & name, const std::string & spec, int64_t ntok) {
    Lauf L{}; L.be = be; L.ntok = ntok; L.nused = 8;
    auto teile_ = teile(spec, '+');
    std::vector<Datei> ds; for (auto & t : teile_) ds.push_back(lade(dir + "/" + name + "." + t + ".bin"));
    int64_t K = ds[0].ne[0], N = ds[0].ne[1]; L.E = ds[0].ne[2];
    bool down = name.find("down") != std::string::npos;
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    L.ctx = ggml_init(ip);
    std::vector<ggml_tensor *> ws;
    for (auto & d : ds) ws.push_back(ggml_new_tensor_3d(L.ctx, d.typ, K, N, L.E));
    L.x = ggml_new_tensor_3d(L.ctx, GGML_TYPE_F32, K, down ? L.nused : 1, ntok);
    L.ids = ggml_new_tensor_2d(L.ctx, GGML_TYPE_I32, L.nused, ntok);
    L.buf = ggml_backend_alloc_ctx_tensors(L.ctx, be);
    if (!L.buf) die("buffer (VRAM?)");
    for (size_t j = 0; j < ds.size(); j++) { ggml_backend_tensor_set(ws[j], ds[j].d.data(), 0, ds[j].d.size()); L.wbytes += ds[j].d.size(); }
    std::vector<float> xv(ggml_nelements(L.x)); std::mt19937 rng(1); std::normal_distribution<float> nd;
    for (auto & v : xv) v = nd(rng);
    ggml_backend_tensor_set(L.x, xv.data(), 0, ggml_nbytes(L.x));
    ggml_init_params gp = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * gc = ggml_init(gp);
    ggml_tensor * y = nullptr;
    for (auto * w : ws) { ggml_tensor * t = ggml_mul_mat_id(gc, w, L.x, L.ids); y = y ? ggml_add(gc, y, t) : t; }
    L.gf = ggml_new_graph(gc); ggml_build_forward_expand(L.gf, y);
    L.ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!ggml_gallocr_alloc_graph(L.ga, L.gf)) die("gallocr");
    return L;
}

static void setze_ids(Lauf & L, std::mt19937 & rng) {
    std::vector<int32_t> v(L.nused * L.ntok); std::vector<int32_t> alle(L.E);
    for (int i = 0; i < L.E; i++) alle[i] = i;
    for (int64_t t = 0; t < L.ntok; t++) {
        for (int j = 0; j < L.nused; j++) std::swap(alle[j], alle[j + rng() % (L.E - j)]);
        for (int j = 0; j < L.nused; j++) v[t * L.nused + j] = alle[j];
    }
    ggml_backend_tensor_set(L.ids, v.data(), 0, v.size() * 4);
}

// cpu|gpu DIR NAME NTOK REPS SPEC...  (DIR = quantized tensors written by "quant")
static int cmd_rechnen(bool gpu, int argc, char ** argv) {
    if (argc < 6) die("cpu|gpu DIR NAME NTOK REPS SPEC...");
    std::string dir = argv[1], name = argv[2]; int64_t ntok = atoll(argv[3]); int reps = atoi(argv[4]);
    ggml_backend_t be = gpu ? ggml_backend_cuda_init(0) : ggml_backend_cpu_init();
    int nth = getenv("TT_THREADS") ? atoi(getenv("TT_THREADS")) : 10;
    if (!gpu) ggml_backend_cpu_set_n_threads(be, nth);
    for (int i = 5; i < argc; i++) {
        Lauf L = baue(be, dir, name, argv[i], ntok);
        std::mt19937 rng(42);
        std::vector<double> ts;
        // bytes read per call: for small ntok only the selected experts
        double anteil_exp = std::min(1.0, (double) L.nused * ntok / L.E);  // rough upper bound
        for (int r = 0; r < reps + 3; r++) {
            setze_ids(L, rng); ggml_backend_synchronize(be);
            auto a = clk::now();
            ggml_backend_graph_compute(be, L.gf); ggml_backend_synchronize(be);
            if (r >= 3) ts.push_back(sek(a, clk::now()));
        }
        double m = median(ts), p10, p90;
        std::sort(ts.begin(), ts.end()); p10 = ts[ts.size() / 10]; p90 = ts[ts.size() * 9 / 10];
        double gelesen = ntok == 1 ? (double) L.wbytes * L.nused / L.E : L.wbytes * anteil_exp;
        printf("%s\t%s\t%s\tntok=%ld\tbytes=%zu\tmedian_us=%.1f\tp10_us=%.1f\tp90_us=%.1f\tgbs_gelesen=%.2f\tthreads=%d\n",
               gpu ? "gpu" : "cpu", name.c_str(), argv[i], ntok, L.wbytes, m * 1e6, p10 * 1e6, p90 * 1e6, gelesen / m / 1e9, gpu ? 0 : nth);
        fflush(stdout);
        ggml_gallocr_free(L.ga); ggml_backend_buffer_free(L.buf); ggml_free(L.ctx);
    }
    ggml_backend_free(be);
    return 0;
}

// h2d DIR NAME REPS SPEC...  whole tensor host->VRAM, pinned (CUDA host buffer) and pageable (malloc)
static int cmd_h2d(int argc, char ** argv) {
    std::string dir = argv[1], name = argv[2]; int reps = atoi(argv[3]);
    ggml_backend_t be = ggml_backend_cuda_init(0);
    for (int i = 4; i < argc; i++) {
        Datei d = lade(dir + "/" + name + "." + argv[i] + ".bin");
        size_t n = d.d.size();
        ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
        ggml_context * c = ggml_init(ip);
        ggml_tensor * t = ggml_new_tensor_1d(c, GGML_TYPE_I8, n);
        ggml_backend_buffer_t vb = ggml_backend_alloc_ctx_tensors(c, be);
        ggml_backend_buffer_t hb = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_host_buffer_type(), n);
        void * pin = ggml_backend_buffer_get_base(hb); memcpy(pin, d.d.data(), n);
        for (int modus = 0; modus < 2; modus++) {
            const void * src = modus == 0 ? pin : (const void *) d.d.data();
            std::vector<double> ts;
            for (int r = 0; r < reps + 2; r++) {
                auto a = clk::now(); ggml_backend_tensor_set(t, src, 0, n); ggml_backend_synchronize(be);
                if (r >= 2) ts.push_back(sek(a, clk::now()));
            }
            double m = median(ts);
            printf("h2d\t%s\t%s\t%s\tbytes=%zu\tmedian_us=%.1f\tgbs=%.2f\n", name.c_str(), argv[i], modus == 0 ? "gepinnt" : "pageable", n, m * 1e6, n / m / 1e9);
            fflush(stdout);
        }
        ggml_backend_buffer_free(hb); ggml_backend_buffer_free(vb); ggml_free(c);
    }
    ggml_backend_free(be);
    return 0;
}

// ssd FILE TOTAL_MIB  - O_DIRECT, block sizes 0.25/1/4/16 MiB, 1/2/4/8 threads, TOTAL_MIB each
static int cmd_ssd(int argc, char ** argv) {
    const char * pfad = argv[1]; size_t gesamt = (size_t) atoll(argv[2]) << 20;
    int fd0 = open(pfad, O_RDONLY); off_t groesse = lseek(fd0, 0, SEEK_END); close(fd0);
    size_t bs_liste[] = { 256 << 10, 1 << 20, 4 << 20, 16 << 20 };
    int th_liste[] = { 1, 2, 4, 8 };
    std::mt19937_64 rng(7);
    for (size_t bs : bs_liste) for (int nth : th_liste) {
        size_t pro = gesamt / nth;
        // random start offset per run so no run benefits from the previous one (O_DIRECT bypasses the page cache anyway)
        off_t basis = (off_t) ((rng() % (uint64_t) (groesse - gesamt - (64 << 20))) & ~((off_t) (16 << 20) - 1));
        auto a = clk::now();
        std::vector<std::thread> th;
        for (int t = 0; t < nth; t++) th.emplace_back([&, t] {
            int fd = open(pfad, O_RDONLY | O_DIRECT);
            void * b; if (posix_memalign(&b, 4096, bs)) die("align");
            for (size_t o = 0; o < pro; o += bs) if (pread(fd, b, bs, basis + t * pro + o) != (ssize_t) bs) die("pread");
            free(b); close(fd);
        });
        for (auto & x : th) x.join();
        double s = sek(a, clk::now());
        printf("ssd\tblock_kib=%zu\tthreads=%d\tbytes=%zu\tsek=%.3f\tgbs=%.3f\n", bs >> 10, nth, pro * nth, s, pro * nth / s / 1e9);
        fflush(stdout);
    }
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) die("tt quant|cpu|gpu|h2d|ssd ...");
    std::string c = argv[1];
    if (c == "quant") return cmd_quant(argc - 1, argv + 1);
    if (c == "cpu") return cmd_rechnen(false, argc - 1, argv + 1);
    if (c == "gpu") return cmd_rechnen(true, argc - 1, argv + 1);
    if (c == "h2d") return cmd_h2d(argc - 1, argv + 1);
    if (c == "ssd") return cmd_ssd(argc - 1, argv + 1);
    die("unknown subcommand");
}
