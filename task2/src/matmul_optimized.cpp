// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

#define BLK 16

void matmul_optimized(const float *A, const float *B, float *C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your best combined implementation.
    for (int bi = 0; bi < M / BLK; bi++) {
        for (int bj = 0; bj < N / BLK; bj++) {
            for (int i = 0; i < BLK; i++) {
                for (int j = 0; j < BLK; j++) {
                    C[(bi * BLK + i) * ldc + bj * BLK + j] = 0.0f;
                    for (int bk = 0; bk < K / BLK; bk++) {
                        __m256 vecAdd = _mm256_setzero_ps();
                        const float *a = A + (bi * BLK + i) * lda + bk * BLK;
                        const float *b = B + (bj * BLK + j) * ldb + bk * BLK;

                        for (int p = 0; p < BLK; p += 8)
                            vecAdd = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), vecAdd);

                        __m128 low = _mm256_castps256_ps128(vecAdd);
                        __m128 high = _mm256_extractf128_ps(vecAdd, 1);
                        __m128 sum = _mm_add_ps(low, high);

                        __m128 sum2 = _mm_movehl_ps(sum, sum);
                        __m128 sum3 = _mm_add_ps(sum, sum2);
                        __m128 sum4 = _mm_movehdup_ps(sum3);
                        __m128 finalVal = _mm_add_ps(sum3, sum4);

                        C[(bi * BLK + i) * ldc + bj * BLK + j] += _mm_cvtss_f32(finalVal);
                    }
                }
            }
        }
    }
}
