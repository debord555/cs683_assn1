// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox
#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    /*
     * Optimized for K = 3.
     *
     * Loop structure:
     *
     *     oy
     *       ox (unrolled by 8)
     *          ky
     *             kx
     *
     * Instead of calculating one output at a time,
     * calculate 8 neighboring output pixels together.
     */

    if (K == 3) {

        const float k00 = ker[0];
        const float k01 = ker[1];
        const float k02 = ker[2];

        const float k10 = ker[3];
        const float k11 = ker[4];
        const float k12 = ker[5];

        const float k20 = ker[6];
        const float k21 = ker[7];
        const float k22 = ker[8];

        for (int oy = 0; oy < H; ++oy) {

            const float* r0 = in + oy * in_stride;
            const float* r1 = r0 + in_stride;
            const float* r2 = r1 + in_stride;

            float* out_row = out + oy * W;

            int ox = 0;

            // Unroll output loop by 8
            for (; ox + 7 < W; ox += 8) {

                float acc0 = 0.0f;
                float acc1 = 0.0f;
                float acc2 = 0.0f;
                float acc3 = 0.0f;
                float acc4 = 0.0f;
                float acc5 = 0.0f;
                float acc6 = 0.0f;
                float acc7 = 0.0f;

                // ky = 0
                acc0 += r0[ox]     * k00;
                acc1 += r0[ox + 1] * k00;
                acc2 += r0[ox + 2] * k00;
                acc3 += r0[ox + 3] * k00;
                acc4 += r0[ox + 4] * k00;
                acc5 += r0[ox + 5] * k00;
                acc6 += r0[ox + 6] * k00;
                acc7 += r0[ox + 7] * k00;

                acc0 += r0[ox + 1] * k01;
                acc1 += r0[ox + 2] * k01;
                acc2 += r0[ox + 3] * k01;
                acc3 += r0[ox + 4] * k01;
                acc4 += r0[ox + 5] * k01;
                acc5 += r0[ox + 6] * k01;
                acc6 += r0[ox + 7] * k01;
                acc7 += r0[ox + 8] * k01;

                acc0 += r0[ox + 2] * k02;
                acc1 += r0[ox + 3] * k02;
                acc2 += r0[ox + 4] * k02;
                acc3 += r0[ox + 5] * k02;
                acc4 += r0[ox + 6] * k02;
                acc5 += r0[ox + 7] * k02;
                acc6 += r0[ox + 8] * k02;
                acc7 += r0[ox + 9] * k02;

                // ky = 1
                acc0 += r1[ox]     * k10;
                acc1 += r1[ox + 1] * k10;
                acc2 += r1[ox + 2] * k10;
                acc3 += r1[ox + 3] * k10;
                acc4 += r1[ox + 4] * k10;
                acc5 += r1[ox + 5] * k10;
                acc6 += r1[ox + 6] * k10;
                acc7 += r1[ox + 7] * k10;

                acc0 += r1[ox + 1] * k11;
                acc1 += r1[ox + 2] * k11;
                acc2 += r1[ox + 3] * k11;
                acc3 += r1[ox + 4] * k11;
                acc4 += r1[ox + 5] * k11;
                acc5 += r1[ox + 6] * k11;
                acc6 += r1[ox + 7] * k11;
                acc7 += r1[ox + 8] * k11;

                acc0 += r1[ox + 2] * k12;
                acc1 += r1[ox + 3] * k12;
                acc2 += r1[ox + 4] * k12;
                acc3 += r1[ox + 5] * k12;
                acc4 += r1[ox + 6] * k12;
                acc5 += r1[ox + 7] * k12;
                acc6 += r1[ox + 8] * k12;
                acc7 += r1[ox + 9] * k12;

                // ky = 2
                acc0 += r2[ox]     * k20;
                acc1 += r2[ox + 1] * k20;
                acc2 += r2[ox + 2] * k20;
                acc3 += r2[ox + 3] * k20;
                acc4 += r2[ox + 4] * k20;
                acc5 += r2[ox + 5] * k20;
                acc6 += r2[ox + 6] * k20;
                acc7 += r2[ox + 7] * k20;

                acc0 += r2[ox + 1] * k21;
                acc1 += r2[ox + 2] * k21;
                acc2 += r2[ox + 3] * k21;
                acc3 += r2[ox + 4] * k21;
                acc4 += r2[ox + 5] * k21;
                acc5 += r2[ox + 6] * k21;
                acc6 += r2[ox + 7] * k21;
                acc7 += r2[ox + 8] * k21;

                acc0 += r2[ox + 2] * k22;
                acc1 += r2[ox + 3] * k22;
                acc2 += r2[ox + 4] * k22;
                acc3 += r2[ox + 5] * k22;
                acc4 += r2[ox + 6] * k22;
                acc5 += r2[ox + 7] * k22;
                acc6 += r2[ox + 8] * k22;
                acc7 += r2[ox + 9] * k22;

                out_row[ox]     = acc0;
                out_row[ox + 1] = acc1;
                out_row[ox + 2] = acc2;
                out_row[ox + 3] = acc3;
                out_row[ox + 4] = acc4;
                out_row[ox + 5] = acc5;
                out_row[ox + 6] = acc6;
                out_row[ox + 7] = acc7;
            }

            // Remaining columns
            for (; ox < W; ++ox) {

                float acc = 0.0f;

                for (int ky = 0; ky < K; ++ky) {
                    for (int kx = 0; kx < K; ++kx) {

                        acc +=
                            in[(oy + ky) * in_stride + (ox + kx)]
                            * ker[ky * K + kx];
                    }
                }

                out_row[ox] = acc;
            }
        }

        return;
    }

    // Generic fallback for K != 3
    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {

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