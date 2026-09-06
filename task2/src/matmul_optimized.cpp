// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <cstring>
#include <immintrin.h>

#include "matmul.h"

#define BLK 96

static inline float hsum256(__m256 v) {
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);
    __m128 sum2 = _mm_movehl_ps(sum, sum);
    __m128 sum3 = _mm_add_ps(sum, sum2);
    __m128 suM = _mm_movehdup_ps(sum3);
    return _mm_cvtss_f32(_mm_add_ps(sum3, suM));
}

void matmul_optimized(const float *A, const float *B, float *C,
                      int M, int N, int K, int lda, int ldb, int ldc) {

    memset(C, 0, sizeof(float) * M * ldc);
    for (int bi = 0; bi < M; bi += BLK) {
        for (int bj = 0; bj < N; bj += BLK) {

            int ei = bi + BLK < M ? bi + BLK : M;
            int ei4 = ei - (ei % 4);
            int ej = bj + BLK < N ? bj + BLK : N;
            int ej2 = ej - (ej % 3);
            for (int i = bi; i < ei4; i += 4) {

                const float *a0 = A + static_cast<long>(i + 0) * lda;
                const float *a1 = A + static_cast<long>(i + 1) * lda;
                const float *a2 = A + static_cast<long>(i + 2) * lda;
                const float *a3 = A + static_cast<long>(i + 3) * lda;

                for (int j = bj; j < ej2; j += 3) {

                    __m256 acc00 = _mm256_setzero_ps();
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc02 = _mm256_setzero_ps();
                    __m256 acc10 = _mm256_setzero_ps();
                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc12 = _mm256_setzero_ps();
                    __m256 acc20 = _mm256_setzero_ps();
                    __m256 acc21 = _mm256_setzero_ps();
                    __m256 acc22 = _mm256_setzero_ps();
                    __m256 acc30 = _mm256_setzero_ps();
                    __m256 acc31 = _mm256_setzero_ps();
                    __m256 acc32 = _mm256_setzero_ps();

                    const float *b0 = B + static_cast<long>(j + 0) * ldb;
                    const float *b1 = B + static_cast<long>(j + 1) * ldb;
                    const float *b2 = B + static_cast<long>(j + 2) * ldb;

                    // block size must always be multiple of 8 for this to work
                    for (int p = 0; p < K; p += 8) {

                        __m256 va0 = _mm256_loadu_ps(a0 + p);
                        __m256 va1 = _mm256_loadu_ps(a1 + p);
                        __m256 va2 = _mm256_loadu_ps(a2 + p);
                        __m256 va3 = _mm256_loadu_ps(a3 + p);

                        __m256 vb0 = _mm256_loadu_ps(b0 + p);
                        __m256 vb1 = _mm256_loadu_ps(b1 + p);
                        __m256 vb2 = _mm256_loadu_ps(b2 + p);

                        acc00 = _mm256_fmadd_ps(va0, vb0, acc00);
                        acc01 = _mm256_fmadd_ps(va0, vb1, acc01);
                        acc02 = _mm256_fmadd_ps(va0, vb2, acc02);
                        acc10 = _mm256_fmadd_ps(va1, vb0, acc10);
                        acc11 = _mm256_fmadd_ps(va1, vb1, acc11);
                        acc12 = _mm256_fmadd_ps(va1, vb2, acc12);
                        acc20 = _mm256_fmadd_ps(va2, vb0, acc20);
                        acc21 = _mm256_fmadd_ps(va2, vb1, acc21);
                        acc22 = _mm256_fmadd_ps(va2, vb2, acc22);
                        acc30 = _mm256_fmadd_ps(va3, vb0, acc30);
                        acc31 = _mm256_fmadd_ps(va3, vb1, acc31);
                        acc32 = _mm256_fmadd_ps(va3, vb2, acc32);
                    }

                    float s00 = hsum256(acc00);
                    float s01 = hsum256(acc01);
                    float s02 = hsum256(acc02);
                    float s10 = hsum256(acc10);
                    float s11 = hsum256(acc11);
                    float s12 = hsum256(acc12);
                    float s20 = hsum256(acc20);
                    float s21 = hsum256(acc21);
                    float s22 = hsum256(acc22);
                    float s30 = hsum256(acc30);
                    float s31 = hsum256(acc31);
                    float s32 = hsum256(acc32);

                    C[static_cast<long>(i + 0) * ldc + (j + 0)] += s00;
                    C[static_cast<long>(i + 0) * ldc + (j + 1)] += s01;
                    C[static_cast<long>(i + 0) * ldc + (j + 2)] += s02;
                    C[static_cast<long>(i + 1) * ldc + (j + 0)] += s10;
                    C[static_cast<long>(i + 1) * ldc + (j + 1)] += s11;
                    C[static_cast<long>(i + 1) * ldc + (j + 2)] += s12;
                    C[static_cast<long>(i + 2) * ldc + (j + 0)] += s20;
                    C[static_cast<long>(i + 2) * ldc + (j + 1)] += s21;
                    C[static_cast<long>(i + 2) * ldc + (j + 2)] += s22;
                    C[static_cast<long>(i + 3) * ldc + (j + 0)] += s30;
                    C[static_cast<long>(i + 3) * ldc + (j + 1)] += s31;
                    C[static_cast<long>(i + 3) * ldc + (j + 2)] += s32;
                }
            }

            for (int i = bi; i < ei4; i++) {
                const float *a = A + static_cast<long>(i) * lda;
                for (int j = ej2; j < ej; j++) {
                    const float *b = B + static_cast<long>(j) * ldb;
                    __m256 acc = _mm256_setzero_ps();
                    for (int p = 0; p < K; p += 8) {
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
                    }
                    C[i * ldc + j] += hsum256(acc);
                }
            }

            for (int i = ei4; i < ei; i++) {
                const float *a = A + static_cast<long>(i) * lda;
                for (int j = bj; j < ej; j++) {
                    const float *b = B + static_cast<long>(j) * ldb;
                    __m256 acc = _mm256_setzero_ps();
                    for (int p = 0; p < K; p += 8) {
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc);
                    }
                    C[i * ldc + j] += hsum256(acc);
                }
            }
        }
    }
}
