// conv_tile.cpp  STAGE 3: CACHE TILING
 #ifndef TILE_SIZE
#define TILE_SIZE 128
#endif
#include "convolution.h"
void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K)
{
  const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oyy = 0; oyy < H; oyy += TILE_SIZE) {

        for (int oxx = 0; oxx < W; oxx += TILE_SIZE) {

            int oy_end = oyy + TILE_SIZE;
            if (oy_end > H)
                oy_end = H;

            int ox_end = oxx + TILE_SIZE;
            if (ox_end > W)
                ox_end = W;

            for (int oy = oyy; oy < oy_end; ++oy) {

                for (int ox = oxx; ox < ox_end; ++ox) {

                    float acc = 0.0f;

                    for (int ky = 0; ky < K; ++ky) {

                        for (int kx = 0; kx < K; ++kx) {

                            acc +=
                                in[(oy + ky) * in_stride + (ox + kx)]
                                * ker[ky * K + kx];
                        }
                    }

                    out[oy * W + ox] = acc;
                }
            }
        }
    }
   
}
               
