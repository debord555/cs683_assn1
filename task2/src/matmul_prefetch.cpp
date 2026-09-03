// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#define BLK 32

#define PREFETCH_DISTANCE 16


void matmul_prefetch(const float *A, const float *B, float *C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    /*for (int bi = 0; bi < M / BLK; bi++) {
        for (int bj = 0; bj < N / BLK; bj++) {
            for (int i = 0; i < BLK; i++) {
                for (int j = 0; j < BLK; j++) {
                    C[(bi * BLK + i) * ldc + bj * BLK + j] = 0.0f;
                    for (int bk = 0; bk < K / BLK; bk++) {

                        float acc = 0.0f;

                        for (int p = 0; p < BLK; p++) {
                            acc += A[(bi * BLK + i) * lda + bk * BLK + p] *
                                   B[(bj * BLK + j) * ldb + bk * BLK + p];
                        }

                        C[(bi * BLK + i) * ldc + bj * BLK + j] += acc;
                    }
                }
            }
        }
    }*/

    for (int bi = 0; bi < M; bi += BLK) {
        int i_end = (bi + BLK < M) ? bi + BLK : M;

        for (int bj = 0; bj < N; bj += BLK) {
            int j_end = (bj + BLK < N) ? bj + BLK : N;

            for (int i = bi; i < i_end; i++) {
                for (int j = bj; j < j_end; j++) {
                    C[(long)i * ldc + j] = 0.0f;
                }
            }

            for (int bk = 0; bk < K; bk += BLK) {
                int k_end = (bk + BLK < K) ? bk + BLK : K;

                for (int i = bi; i < i_end; i++) {
                    const float *a = A + (long)i * lda + bk;

                    for (int j = bj; j < j_end; j++) {
                        const float *b = B + (long)j * ldb + bk;

                        float acc = 0.0f;

                        for (int p = 0; p < k_end - bk; p++) {
                            if (p + PREFETCH_DISTANCE < k_end - bk) {
                                _mm_prefetch(
                                    (const char *)(a + p + PREFETCH_DISTANCE),
                                    _MM_HINT_T0
                                );

                                _mm_prefetch(
                                    (const char *)(b + p + PREFETCH_DISTANCE),
                                    _MM_HINT_T0
                                );
                            }

                            acc += a[p] * b[p];
                        }

                        C[(long)i * ldc + j] += acc;
                    }
                }
            }
        }
    }
}
