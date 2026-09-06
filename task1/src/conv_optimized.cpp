#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        int ox = 0;
        
        // --- 32-WIDE SIMD UNROLLING ---
        // We use 4 YMM registers (v_acc0 to v_acc3). 
        // Each handles 8 floats, totaling 32 pixels per iteration.
        for (; ox <= W - 32; ox += 32) {
            __m256 v_acc0 = _mm256_setzero_ps();
            __m256 v_acc1 = _mm256_setzero_ps();
            __m256 v_acc2 = _mm256_setzero_ps();
            __m256 v_acc3 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    // Broadcast the single kernel weight to all 8 lanes
                    __m256 v_ker = _mm256_set1_ps(ker[ky * K + kx]);

                    // Load 32 contiguous pixels (4 blocks of 8)
                    __m256 v_in0 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + 0 + kx)]);
                    __m256 v_in1 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + 8 + kx)]);
                    __m256 v_in2 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + 16 + kx)]);
                    __m256 v_in3 = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + 24 + kx)]);

                    // Parallel FMA: acc = (in * ker) + acc
                    v_acc0 = _mm256_fmadd_ps(v_in0, v_ker, v_acc0);
                    v_acc1 = _mm256_fmadd_ps(v_in1, v_ker, v_acc1);
                    v_acc2 = _mm256_fmadd_ps(v_in2, v_ker, v_acc2);
                    v_acc3 = _mm256_fmadd_ps(v_in3, v_ker, v_acc3);
                }
            }
            // Store results back to memory
            _mm256_storeu_ps(&out[oy * W + ox + 0],  v_acc0);
            _mm256_storeu_ps(&out[oy * W + ox + 8],  v_acc1);
            _mm256_storeu_ps(&out[oy * W + ox + 16], v_acc2);
            _mm256_storeu_ps(&out[oy * W + ox + 24], v_acc3);
        }

        // --- 8-WIDE TAIL ---
        // Handles remaining pixels in blocks of 8
        for (; ox <= W - 8; ox += 8) {
            __m256 v_acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    __m256 v_ker = _mm256_set1_ps(ker[ky * K + kx]);
                    __m256 v_in = _mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    v_acc = _mm256_fmadd_ps(v_in, v_ker, v_acc);
                }
            }
            _mm256_storeu_ps(&out[oy * W + ox], v_acc);
        }


    }}
