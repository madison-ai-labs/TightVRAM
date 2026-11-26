// STREAM-like memory bandwidth test (copy, triad), OpenMP. 3 arrays of 1 GiB each.
// Input: none (threads via OMP_NUM_THREADS). Output: one line with copy/triad GB/s.
// Build/run: gcc -O3 -march=native -fopenmp stream.c -o stream ; OMP_NUM_THREADS=10 ./stream
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define N (128L*1024*1024)   /* 128 Mi doubles = 1 GiB per array */
#define REPS 10
int main(void){
  double *a=malloc(N*8),*b=malloc(N*8),*c=malloc(N*8);
  #pragma omp parallel for
  for(long i=0;i<N;i++){a[i]=1;b[i]=2;c[i]=0;}
  double best_copy=1e9,best_triad=1e9;
  for(int r=0;r<REPS;r++){
    double t=omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<N;i++) c[i]=a[i];
    t=omp_get_wtime()-t; if(t<best_copy)best_copy=t;
    t=omp_get_wtime();
    #pragma omp parallel for
    for(long i=0;i<N;i++) a[i]=b[i]+3.0*c[i];
    t=omp_get_wtime()-t; if(t<best_triad)best_triad=t;
  }
  printf("threads=%d copy=%.1f GB/s triad=%.1f GB/s (check %.1f)\n", omp_get_max_threads(),
    2.0*N*8/best_copy/1e9, 3.0*N*8/best_triad/1e9, a[N/2]);
  return 0;
}
