#include "matmul.h"
#include <immintrin.h>

#define SIMD_256

#ifdef SIMD_256
#define SIMD_REG __m256
#define SIMD_SETZERO _mm256_setzero_ps()
#define SIMD_LOAD(c) _mm256_loadu_ps(c)
#define SIMD_FMA(a, b, c) _mm256_fmadd_ps(a, b, c)
#define SIMD_HSUM(x) hsum256(x)
#define SIMD_STRIDE 8
#endif

#ifdef SIMD_128
#define SIMD_REG __m128
#define SIMD_SETZERO _mm_setzero_ps()
#define SIMD_LOAD(c) _mm_loadu_ps(c)
#define SIMD_FMA(a, b, c) _mm_fmadd_ps(a, b, c)
#define SIMD_HSUM(x) hsum128(x)
#define SIMD_STRIDE 4
#endif

static inline float hsum128(__m128 v) {
    __m128 sum2 = _mm_movehl_ps(v, v);
    __m128 sum3 = _mm_add_ps(v, sum2);
    __m128 sum4 = _mm_movehdup_ps(sum3);
    return _mm_cvtss_f32(_mm_add_ps(sum3, sum4));
}

static inline float hsum256(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 sum2 = _mm_movehl_ps(sum, sum);
    __m128 sum3 = _mm_add_ps(sum, sum2);
    __m128 suM = _mm_movehdup_ps(sum3);
    return _mm_cvtss_f32(_mm_add_ps(sum3, suM));
}

void matmul_simd(const float *A, const float *B, float *C,
                     int M, int N, int K, int lda, int ldb, int ldc) {

    int M4 = M - (M % 4);
    int N2 = N - (N % 2);

    for (int i = 0; i < M4; i += 4) {
        for (int j = 0; j < N2; j += 2) {
            SIMD_REG acc00 = SIMD_SETZERO;
            SIMD_REG acc01 = SIMD_SETZERO;
            SIMD_REG acc10 = SIMD_SETZERO;
            SIMD_REG acc11 = SIMD_SETZERO;
            SIMD_REG acc20 = SIMD_SETZERO;
            SIMD_REG acc21 = SIMD_SETZERO;
            SIMD_REG acc30 = SIMD_SETZERO;
            SIMD_REG acc31 = SIMD_SETZERO;

            const float *a0 = A + static_cast<long>(i + 0) * lda;
            const float *a1 = A + static_cast<long>(i + 1) * lda;
            const float *a2 = A + static_cast<long>(i + 2) * lda;
            const float *a3 = A + static_cast<long>(i + 3) * lda;

            const float *b0 = B + static_cast<long>(j + 0) * ldb;
            const float *b1 = B + static_cast<long>(j + 1) * ldb;

            for (int p = 0; p < K; p += SIMD_STRIDE) {
                SIMD_REG va0 = SIMD_LOAD(a0 + p);
                SIMD_REG va1 = SIMD_LOAD(a1 + p);
                SIMD_REG va2 = SIMD_LOAD(a2 + p);
                SIMD_REG va3 = SIMD_LOAD(a3 + p);

                SIMD_REG vb0 = SIMD_LOAD(b0 + p);
                SIMD_REG vb1 = SIMD_LOAD(b1 + p);

                acc00 = SIMD_FMA(va0, vb0, acc00);
                acc01 = SIMD_FMA(va0, vb1, acc01);
                acc10 = SIMD_FMA(va1, vb0, acc10);
                acc11 = SIMD_FMA(va1, vb1, acc11);
                acc20 = SIMD_FMA(va2, vb0, acc20);
                acc21 = SIMD_FMA(va2, vb1, acc21);
                acc30 = SIMD_FMA(va3, vb0, acc30);
                acc31 = SIMD_FMA(va3, vb1, acc31);
            }

            float s00 = SIMD_HSUM(acc00);
            float s01 = SIMD_HSUM(acc01);
            float s10 = SIMD_HSUM(acc10);
            float s11 = SIMD_HSUM(acc11);
            float s20 = SIMD_HSUM(acc20);
            float s21 = SIMD_HSUM(acc21);
            float s30 = SIMD_HSUM(acc30);
            float s31 = SIMD_HSUM(acc31);

            C[static_cast<long>(i + 0) * ldc + (j + 0)] = s00;
            C[static_cast<long>(i + 0) * ldc + (j + 1)] = s01;
            C[static_cast<long>(i + 1) * ldc + (j + 0)] = s10;
            C[static_cast<long>(i + 1) * ldc + (j + 1)] = s11;
            C[static_cast<long>(i + 2) * ldc + (j + 0)] = s20;
            C[static_cast<long>(i + 2) * ldc + (j + 1)] = s21;
            C[static_cast<long>(i + 3) * ldc + (j + 0)] = s30;
            C[static_cast<long>(i + 3) * ldc + (j + 1)] = s31;
        }
    }

    for (int i = 0; i < M4; i++) {
        for (int j = N2; j < N; j++) {
            float acc = 0.0f;
            for (int p = 0; p < K; p++) {
                acc += A[i * lda + p] * B[j * ldb + p];
            }
            C[i * ldc + j] = acc;
        }
    }

    for (int i = M4; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int p = 0; p < K; p++) {
                acc += A[i * lda + p] * B[j * ldb + p];
            }
            C[i * ldc + j] = acc;
        }
    }
}