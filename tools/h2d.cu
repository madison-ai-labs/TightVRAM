// Host-to-device copy bandwidth: pinned vs pageable memory, then small blocks (expert-sized) from pinned memory.
// Input: none. Output: GB/s lines on stdout. Build/run: nvcc -O2 h2d.cu -o h2d && ./h2d
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
int main(){
  const size_t N = 256ull<<20; const int R = 20;
  void *d; cudaMalloc(&d, N);
  void *hp; cudaMallocHost(&hp, N);
  void *hq = malloc(N);
  memset(hq, 1, N); memset(hp, 1, N);
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int mode=0; mode<2; mode++){
    void *h = mode ? hq : hp;
    for (int w=0; w<3; w++) cudaMemcpy(d,h,N,cudaMemcpyHostToDevice);
    for (int run=0; run<3; run++){
      cudaEventRecord(a);
      for (int i=0;i<R;i++) cudaMemcpyAsync(d,h,N,cudaMemcpyHostToDevice);
      cudaEventRecord(b); cudaEventSynchronize(b);
      float ms; cudaEventElapsedTime(&ms,a,b);
      printf("%s run %d: %.2f GB/s\n", mode?"pageable":"pinned  ", run, (double)N*R/ms/1e6);
    }
  }
  // small blocks like one expert (~0.6 MB per matrix)
  for (size_t blk : {size_t(640<<10), size_t(2<<20), size_t(16<<20)}){
    int n = N/blk;
    cudaEventRecord(a);
    for (int i=0;i<n;i++) cudaMemcpyAsync((char*)d+i*blk,(char*)hp+i*blk,blk,cudaMemcpyHostToDevice);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms; cudaEventElapsedTime(&ms,a,b);
    printf("pinned blk %6zu KiB: %.2f GB/s\n", blk>>10, (double)N/ms/1e6);
  }
  printf("err: %s\n", cudaGetErrorString(cudaGetLastError()));
}
