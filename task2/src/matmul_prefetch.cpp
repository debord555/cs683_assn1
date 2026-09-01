// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#define BLK 64

void matmul_prefetch(const float *A, const float *B, float *C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.
    for (int bi = 0; bi < M / BLK; bi++) {
        for (int bj = 0; bj < N / BLK; bj++) {
            for (int i = 0; i < BLK; i++) {
                for (int j = 0; j < BLK; j++) {
                    C[(bi * BLK + i) * ldc + bj * BLK + j] = 0.0;
                }
            }
            for (int bk = 0; bk < K / BLK; bk++) {
                for (int i = 0; i < BLK; i++) {
                    for (int j = 0; j < BLK; j++) {
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
    }
}
