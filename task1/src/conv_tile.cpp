#include "convolution.h"
#include <algorithm> 

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int TH = 32; 
    const int TW = 32;

    for (int ty = 0; ty < H; ty += TH) {
        for (int tx = 0; tx < W; tx += TW) {
            
            for (int oy = ty; oy < std::min(ty + TH, H); ++oy) {
                for (int ox = tx; ox < std::min(tx + TW, W); ++ox) {
                    
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        for (int kx = 0; kx < K; ++kx) {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                    
                }
            }
        }
    }
}
