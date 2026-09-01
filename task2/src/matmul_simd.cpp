// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float *A, const float *B, float *C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    // TODO(student): replace this placeholder with your register-tiled AVX2 implementation.
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            const float *a = A + static_cast<long>(i) * lda;
            const float *b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; p += 8) {
                __m256 result = _mm256_mul_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p));

                __m128 low = _mm256_castps256_ps128(result);
                __m128 high = _mm256_extractf128_ps(result, 1);
                __m128 sum = _mm_add_ps(low, high);

                __m128 sum2 = _mm_movehl_ps(sum, sum);
                __m128 sum3 = _mm_add_ps(sum, sum2);
                __m128 sum4 = _mm_movehdup_ps(sum3);
                __m128 finalVal = _mm_add_ps(sum3, sum4);

                acc += _mm_cvtss_f32(finalVal);
            }

            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
