// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#define BLK 16
#define PREF_DIST 2

void matmul_prefetch(const float *A, const float *B, float *C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.

    for (int bi = 0; bi < M; bi += BLK) {
        for (int bj = 0; bj < N; bj += BLK) {
            for (int i = bi, ei = bi + BLK < M ? bi + BLK : M; i < ei; i++) {
                for (int j = bj, ej = bj + BLK < N ? bj + BLK : N; j < ej; j++) {
                    C[i * ldc + j] = 0.0f;
                    for (int bk = 0; bk < K; bk += BLK) {
                        float acc = 0.0f;

                        for (int p = bk, pe = bk + BLK < K ? bk + BLK : K; p < pe; p++) {
                            acc += A[i * lda + p] *
                                   B[j * ldb + p];
                        }

                        C[i * ldc + j] += acc;
                    }
                }
            }
        }
    }

}
