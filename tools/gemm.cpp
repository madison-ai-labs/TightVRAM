// Dense GEMM via cuBLAS, evidence for the tensor-core rate independent of llama.cpp.
// Three cases: FP16 input with FP32 accumulation (affected by the unlock), TF32 (also affected),
// FP32 without tensor cores (control, must not move). Median of 20 runs after 5 warm-up runs.
// Input: matrix size n (default 4096). Output: n and TFLOPS per case, as one JSON line.
// build: nvcc -O2 gemm.cpp -lcublas -o gemm    usage: gemm [n]
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { auto e_ = (x); if ((int)e_ != 0) { fprintf(stderr, "%s failed: %d\n", #x, (int)e_); exit(1); } } while (0)

static double run(cublasHandle_t h, int n, cublasComputeType_t ct, cublasDataType_t in, size_t sz, void *a, void *b, void *c) {
    const float one = 1.0f, zero = 0.0f;
    std::vector<double> t;
    for (int i = 0; i < 25; i++) {
        CK(cudaDeviceSynchronize());
        auto s = std::chrono::steady_clock::now();
        // beta 0 and FP32 result: accumulation stays FP32.
        CK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, a, in, n, b, in, n, &zero, c, CUDA_R_32F, n, ct, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CK(cudaDeviceSynchronize());
        if (i >= 5) t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count());
    }
    std::sort(t.begin(), t.end());
    (void)sz;
    return 2.0 * n * n * n / t[t.size() / 2] / 1e12;
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 4096;
    size_t cnt = (size_t)n * n;
    std::vector<float> hf(cnt);
    std::vector<half> hh(cnt);
    srand(42);
    for (size_t i = 0; i < cnt; i++) { hf[i] = (float)rand() / RAND_MAX; hh[i] = __float2half(hf[i]); }

    cublasHandle_t h;
    CK(cublasCreate(&h));
    void *af, *bf, *ah, *bh, *c;
    CK(cudaMalloc(&af, cnt * 4)); CK(cudaMalloc(&bf, cnt * 4)); CK(cudaMalloc(&c, cnt * 4));
    CK(cudaMalloc(&ah, cnt * 2)); CK(cudaMalloc(&bh, cnt * 2));
    CK(cudaMemcpy(af, hf.data(), cnt * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(bf, hf.data(), cnt * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ah, hh.data(), cnt * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(bh, hh.data(), cnt * 2, cudaMemcpyHostToDevice));

    printf("n=%d\n", n);
    double fp16 = run(h, n, CUBLAS_COMPUTE_32F, CUDA_R_16F, 2, ah, bh, c);
    double tf32 = run(h, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, 4, af, bf, c);
    double fp32 = run(h, n, CUBLAS_COMPUTE_32F, CUDA_R_32F, 4, af, bf, c);
    printf("{\"n\":%d,\"fp16_acc32_tflops\":%.2f,\"tf32_tflops\":%.2f,\"fp32_tflops\":%.2f}\n", n, fp16, tf32, fp32);
    return 0;
}
