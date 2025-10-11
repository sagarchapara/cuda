#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>
#include <math_constants.h>

__global__ void flash_attn_fwd(
    int B,
    int H,
    int N,
    int D,
    const float *Q,
    const float *K,
    const float *V,
    float *O,
    float scaling_factor,
    const bool *mask,
    float dropout_prob,
    const int M,
    int BC,
    int BR,
    float *l,
    float *m)
{

    int tc = (N + BC - 1) / BC;
    int tr = (N + BR - 1) / BR;

    extern __shared__ float smem[];
    float *Qi = smem;             // BR * D
    float *Ki = Qi + (BR * D);    // BC * D
    float *Vi = Ki + (BC * D);    // BC * D
    float *Mi = Vi + (BC * D);    // BR
    float *Li = Mi + BR;          // BR
    float *Oi = Li + BR;          // BR * D
    float *MiOld = Oi + (BR * D); // BR

    const int batch_offset = blockIdx.z * N * D;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    const int i = blockIdx.x;

    if (i >= tr)
        return;

    // Load Qi and initialize Oi, Mi, Li
    for (int idx = threadIdx.x; idx < BR * D; idx += blockDim.x)
    {
        Qi[idx] = Q[batch_offset + i * BR * D + idx];
        Oi[idx] = 0.0f;
    }

    for (int r = threadIdx.x; r < BR; r += blockDim.x)
    {
        Mi[r] = -CUDART_INF_F;
        Li[r] = 0.0f;
    }

    __syncthreads();

    for (int j = 0; j < tc; j++)
    {
        {
            // Load Kj, Vj
            const int base = batch_offset + j * BC * D;
            for (int idx = threadIdx.x; idx < BC * D; idx += blockDim.x)
            {
                Ki[idx] = K[base + idx];
                Vi[idx] = V[base + idx];
            }
        }

        __syncthreads();

        // Warp-per-row reduction: each warp scans all columns of its assigned rows in 32-wide tiles.
        // Eliminates the multi-stride shared-memory halving + barriers.
        // Assumptions:
        //  - blockDim.x is a multiple of 32.
        //  - BR can be >= num_warps; we iterate rows in a strided fashion per warp.
        //  - Scores are computed on-the-fly; we compute Mi[row] = max over columns.
        // lane/warp_id/num_warps already computed above

        for (int row = warp_id; row < BR; row += num_warps)
        {
            float vmax = -CUDART_INF_F;
            // Iterate over column tiles of width 32.
            for (int base = 0; base < BC; base += 32)
            {
                int col = base + lane;

                if (col < BC)
                {
                    float sum = 0.0f;
                    for (int e = 0; e < D; ++e)
                    {
                        sum += Qi[row * D + e] * Ki[col * D + e];
                    }
                    vmax = fmaxf(vmax, sum * scaling_factor);
                }
            }
// In-warp reduce vmax.
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                float other = __shfl_down_sync(0xffffffff, vmax, offset);
                vmax = fmaxf(vmax, other);
            }
            if (lane == 0)
            {
                MiOld[row] = Mi[row];
                Mi[row] = fmaxf(vmax, Mi[row]);
            }
        }

        __syncthreads();

        // Compute Li[row] = sum_j exp(score(row,j) - Mi[row])
        for (int row = warp_id; row < BR; row += num_warps)
        {
            float lsum = 0.0f;
            const float mval = Mi[row];
            const float mold = MiOld[row];
            const float scale = __expf(-mold + mval);

            for (int dbase = 0; dbase < D; dbase += 32)
            {
                int d = dbase + lane;
                if (d < D)
                    Oi[row * D + d] *= scale;
            }

            for (int base = 0; base < BC; base += 32)
            {
                int col = base + lane;

                float p = 0.0f;
                if (col < BC)
                {
                    float sum = 0.0f;
                    for (int e = 0; e < D; ++e)
                    {
                        sum += Qi[row * D + e] * Ki[col * D + e];
                    }
                    p = __expf(sum * scaling_factor - mval);
                    lsum += p;
                }

                for (int dbase = 0; dbase < D; dbase += 32)
                {
                    int d = dbase + lane;
                    if (d < D && col < BC)
                    {
                        Oi[row * D + d] += p * Vi[col * D + d];
                    }
                }
            }
// In-warp reduce lsum.
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                float other = __shfl_down_sync(0xffffffff, lsum, offset);
                lsum += other;
            }
            if (lane == 0)
                Li[row] = Li[row] * __expf(MiOld[row] - Mi[row]) + lsum;
        }

        __syncthreads();
    }

    // write the Oi to global memory (warp-per-row)
    {
        for (int row = warp_id; row < BR; row += num_warps)
        {
            float Li_row = Li[row];
            for (int cbase = 0; cbase < D; cbase += 32)
            {
                int col = cbase + lane;
                if (col < D)
                {
                    int out_index = batch_offset + i * BR * D + row * D + col;
                    O[out_index] = Oi[row * D + col] / Li_row;
                }
            }
            if (lane == 0)
            {
                l[batch_offset / D + i * BR + row] = Li_row;
            }
        }
    }
}