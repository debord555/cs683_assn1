#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 4) {
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            float acc2 = 0.0f;
            float acc3 = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float k_val = ker[ky * K + kx];
                    
                    acc0 += in[(oy + ky) * in_stride + (ox + 0 + kx)] * k_val;
                    acc1 += in[(oy + ky) * in_stride + (ox + 1 + kx)] * k_val;
                    acc2 += in[(oy + ky) * in_stride + (ox + 2 + kx)] * k_val;
                    acc3 += in[(oy + ky) * in_stride + (ox + 3 + kx)] * k_val;
                }
            }

            out[oy * W + ox + 0] = acc0;
            out[oy * W + ox + 1] = acc1;
            out[oy * W + ox + 2] = acc2;
            out[oy * W + ox + 3] = acc3;
        }
    }
}
