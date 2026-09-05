// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#define BLK 16

#define PREF_DIST 16
#define CLL _MM_HINT_T0

#define MOD_SWP

static inline float hsum256(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 sum2 = _mm_movehl_ps(sum, sum);
    __m128 sum3 = _mm_add_ps(sum, sum2);
    __m128 suM = _mm_movehdup_ps(sum3);
    return _mm_cvtss_f32(_mm_add_ps(sum3, suM));
}

void matmul_swpref(const float *A, const float *B, float *C,
                   int M, int N, int K, int lda, int ldb, int ldc) {
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 vecAdd = _mm256_setzero_ps();
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; p += 8) {

                _mm_prefetch(A + p + PREF_DIST, CLL);
                _mm_prefetch(B + p + PREF_DIST, CLL);
                
                __m256 aval = _mm256_loadu_ps(a + p);
                __m256 bval = _mm256_loadu_ps(b + p);
                vecAdd = _mm256_fmadd_ps(aval, bval, vecAdd);

            }
            C[static_cast<long>(i) * ldc + j] = hsum256(vecAdd);
        }
    }

}

void matmul_blk(const float *A, const float *B, float *C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
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

void matmul_prefetch(const float *A, const float *B, float *C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your cache-blocked SIMD + prefetch
    // implementation.

    #ifdef MOD_BLK
    matmul_blk(A, B, C, M, N, K, lda, ldb, ldc);
    #endif

    #ifdef MOD_SWP
    matmul_swpref(A, B, C, M, N, K, lda, ldb, ldc);
    #endif
    
}
