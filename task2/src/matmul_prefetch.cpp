// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <cstring>
#include <immintrin.h>

#include "matmul.h"

#define BLK 192

#define PREF_DIST 16
#define CLL _MM_HINT_T0

// #define MOD_BLK
// #define MOD_SWP
#define MOD_RSB

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
            const float *a = A + static_cast<long>(i) * lda;
            const float *b = B + static_cast<long>(j) * ldb;
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

void matmul_regtile_simd_blk(const float *A, const float *B, float *C,
                             int M, int N, int K, int lda, int ldb, int ldc) {

    memset(C, 0, sizeof(float) * M * ldc);
    for (int bi = 0; bi < M; bi += BLK) {
        for (int bj = 0; bj < N; bj += BLK) {
            int ei = bi + BLK < M ? bi + BLK : M;
            int ei4 = ei - (ei % 4);
            int ej = bj + BLK < N ? bj + BLK : N;
            int ej2 = ej - (ej % 2);
            for (int i = bi; i < ei4; i += 4) {
                for (int j = bj; j < ej2; j += 2) {
                    for (int bk = 0; bk < K; bk += BLK) {
                        __m256 acc00 = _mm256_setzero_ps();
                        __m256 acc01 = _mm256_setzero_ps();
                        __m256 acc10 = _mm256_setzero_ps();
                        __m256 acc11 = _mm256_setzero_ps();
                        __m256 acc20 = _mm256_setzero_ps();
                        __m256 acc21 = _mm256_setzero_ps();
                        __m256 acc30 = _mm256_setzero_ps();
                        __m256 acc31 = _mm256_setzero_ps();

                        const float *a0 = A + static_cast<long>(i + 0) * lda;
                        const float *a1 = A + static_cast<long>(i + 1) * lda;
                        const float *a2 = A + static_cast<long>(i + 2) * lda;
                        const float *a3 = A + static_cast<long>(i + 3) * lda;

                        const float *b0 = B + static_cast<long>(j + 0) * ldb;
                        const float *b1 = B + static_cast<long>(j + 1) * ldb;

                        // block size must always be multiple of 8 for this to work
                        for (int p = bk, pe = bk + BLK < K ? bk + BLK : K; p < pe; p += 8) {

                            _mm_prefetch((const char *)(a0 + p + PREF_DIST), CLL);
                            _mm_prefetch((const char *)(a1 + p + PREF_DIST), CLL);
                            _mm_prefetch((const char *)(a2 + p + PREF_DIST), CLL);
                            _mm_prefetch((const char *)(a3 + p + PREF_DIST), CLL);

                            _mm_prefetch((const char *)(b0 + p + PREF_DIST), CLL);
                            _mm_prefetch((const char *)(b1 + p + PREF_DIST), CLL);

                            __m256 va0 = _mm256_loadu_ps(a0 + p);
                            __m256 va1 = _mm256_loadu_ps(a1 + p);
                            __m256 va2 = _mm256_loadu_ps(a2 + p);
                            __m256 va3 = _mm256_loadu_ps(a3 + p);

                            __m256 vb0 = _mm256_loadu_ps(b0 + p);
                            __m256 vb1 = _mm256_loadu_ps(b1 + p);

                            acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                            acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                            acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                            acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                            acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                            acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                            acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                            acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                        }

                        float s00 = hsum256(acc00);
                        float s01 = hsum256(acc01);
                        float s10 = hsum256(acc10);
                        float s11 = hsum256(acc11);
                        float s20 = hsum256(acc20);
                        float s21 = hsum256(acc21);
                        float s30 = hsum256(acc30);
                        float s31 = hsum256(acc31);

                        C[static_cast<long>(i + 0) * ldc + (j + 0)] += s00;
                        C[static_cast<long>(i + 0) * ldc + (j + 1)] += s01;
                        C[static_cast<long>(i + 1) * ldc + (j + 0)] += s10;
                        C[static_cast<long>(i + 1) * ldc + (j + 1)] += s11;
                        C[static_cast<long>(i + 2) * ldc + (j + 0)] += s20;
                        C[static_cast<long>(i + 2) * ldc + (j + 1)] += s21;
                        C[static_cast<long>(i + 3) * ldc + (j + 0)] += s30;
                        C[static_cast<long>(i + 3) * ldc + (j + 1)] += s31;
                    }
                }
            }

            for (int i = 0; i < ei4; i++) {
                for (int j = ej2; j < ej; j++) {
                    C[i * ldc + j] = 0.0f;
                    for (int bk = 0; bk < K; bk += BLK) {
                        float acc = 0.0f;
                        const float *a = A + static_cast<long>(i) * lda;
                        const float *b = B + static_cast<long>(j) * ldb;
                        for (int p = bk, pe = bk + BLK < K ? bk + BLK : K; p < pe; p++) {
                            acc += a[p] * b[p];
                        }
                        C[i * ldc + j] += acc;
                    }
                }
            }

            for (int i = ei4; i < ei; i++) {
                for (int j = 0; j < ej; j++) {
                    C[i * ldc + j] = 0.0f;
                    for (int bk = 0; bk < K; bk += BLK) {
                        float acc = 0.0f;
                        const float *a = A + static_cast<long>(i) * lda;
                        const float *b = B + static_cast<long>(j) * ldb;
                        for (int p = bk, pe = bk + BLK < K ? bk + BLK : K; p < pe; p++) {
                            acc += a[p] * b[p];
                        }
                        C[i * ldc + j] += acc;
                    }
                }
            }
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

#ifdef MOD_RSB
    matmul_regtile_simd_blk(A, B, C, M, N, K, lda, ldb, ldc);
#endif
}
