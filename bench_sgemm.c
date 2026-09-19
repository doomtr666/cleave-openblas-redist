// Minimal single-core sgemm throughput probe, matching the exact shape and
// methodology already established for cleave's own matmul measurements
// (doc/backlog.md in the cleave repo): M=32, K=784, N=512 -- the largest
// layer of the real mnist-interop kernel -- timed over many repeated calls
// to amortize fixed overhead, single-threaded for a clean, direct
// comparison against cleave's own reported single-core GFLOP/s figures.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "build/generated/cblas.h"

int main(int argc, char **argv) {
    int M = 32, K = 784, N = 512, iters = 2000, threads = 1;
    if (argc >= 4) { M = atoi(argv[1]); K = atoi(argv[2]); N = atoi(argv[3]); }
    if (argc >= 5) { iters = atoi(argv[4]); }
    if (argc >= 6) { threads = atoi(argv[5]); }
    openblas_set_num_threads(threads);

    float *A = malloc(sizeof(float) * M * K);
    float *B = malloc(sizeof(float) * K * N);
    float *C = malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * K; i++) A[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(i % 5) * 0.1f;
    for (int i = 0; i < M * N; i++) C[i] = 0.0f;

    // Warm up (first call pays one-time JIT-free but still real cache/TLB warmup cost).
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);

    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    for (int it = 0; it < iters; it++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
    }
    timespec_get(&t1, TIME_UTC);

    double elapsed_s = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
    double per_call_ms = (elapsed_s / iters) * 1000.0;
    double flops_per_call = 2.0 * (double)M * (double)K * (double)N;
    double gflops = (flops_per_call * iters) / elapsed_s / 1e9;

    double checksum = 0.0;
    for (int i = 0; i < M * N; i++) checksum += C[i];

    printf("OpenBLAS sgemm M=%d K=%d N=%d, %d iters, %d thread(s)\n", M, K, N, iters, threads);
    printf("total: %.4fs, per-call: %.4fms, throughput: %.2f GFLOP/s\n", elapsed_s, per_call_ms, gflops);
    printf("checksum=%.3f (sanity check, non-zero expected -- C[0] alone is a bad check: it's exactly 0 whenever N is itself a multiple of 5, since B[k*N] = (k*N%%5)*0.1 = 0 for every k in that case, a data-generation artifact, not a real correctness bug)\n", checksum);

    free(A); free(B); free(C);
    return 0;
}
