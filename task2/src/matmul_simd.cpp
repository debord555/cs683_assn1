#include "matmul.h"
#include <immintrin.h>

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

            for (int p = 0; p < K; p += 8) {
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