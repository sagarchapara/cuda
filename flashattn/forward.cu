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

#include <cuda_fp16.h> // __half, __half2, helpers
#include <cuda_bf16.h> // __nv_bfloat16, __nv_bfloat162 (sm80+)
#include <mma.h>

using namespace nvcuda;

#ifndef WM
#define WM 16
#endif
#ifndef WN
#define WN 16
#endif
#ifndef WK
#define WK 16
#endif

// Shared scratch needed per warp: 16*16 floats (1 KB). Provide at kernel launch:
// smem = ... + num_warps * WM * WN * sizeof(float)
inline __device__ float *warp_scratch_ptr(float *warp_scratch_base, int warp_id)
{
    return warp_scratch_base + warp_id * (WM * WN);
}

__global__ void flash_attn_fwd(
    int B,
    int H,
    int N,
    int D,
    const __nv_bfloat16 *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K,
    const __nv_bfloat16 *__restrict__ V,
    float *__restrict__ O,
    float scaling_factor,
    const bool *mask,
    float dropout_prob,
    const int M,
    int BC,
    int BR,
    float *__restrict__ l)
{

    int tc = (N + BC - 1) / BC;
    int tr = (N + BR - 1) / BR;

    extern __shared__ unsigned char smem[];
    __nv_bfloat16 *Qi = reinterpret_cast<__nv_bfloat16 *>(smem); // BR * D
    __nv_bfloat16 *Ki = Qi + (BR * D);                           // BC * D
    __nv_bfloat16 *Vi = Ki + (BC * D);                           // BC * D

    float *warp_scratch_base = reinterpret_cast<float *>(Vi + (BC * D));

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
    }

    __syncthreads();

    // We are giving 16 rows per warp
    for (int j = 0; j < tc; ++j)
    {
        // Flat cooperative copy of K/V tile [BC x D] into shared Ki/Vi
        const int tile_elems = BC * D;
        const int g_base = batch_offset + j * BC * D;
        for (int idx = threadIdx.x; idx < tile_elems; idx += blockDim.x)
        {
            Ki[idx] = K[g_base + idx];
            Vi[idx] = V[g_base + idx];
        }
        __syncthreads();

        // Compute attention score
        for (int row = warp_id * WM; row < BR; row += num_warps * WM)
        {
            float Mi[WM] = {-CUDART_INF_F};
            float Li[WM] = {0.0f};
            if (lane == 0)
            {
#pragma unroll
                for (int r = 0; r < WM; ++r)
                {
                    Mi[r] = -CUDART_INF_F;
                    Li[r] = 0.0f;
                }
            }

            __syncwarp();

            float vmax_rows[WM];

#pragma unroll
            for (int r = 0; r < WM; ++r)
            {
                vmax_rows[r] = -CUDART_INF_F;
            }

            // sweep columns in 16 wide tiles
            for (int col = 0; col < BC; col += WN)
            {
                wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);

                // k dimension in 16 chuncks
                for (int kdim = 0; kdim < D; kdim += WK)
                {
                    // A: 16*16 block from Q, row major
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::precision::bfloat16, wmma::row_major> a_frag;

                    const __nv_bfloat16 *Aptr = Qi + (row * D + kdim);
                    wmma::load_matrix_sync(a_frag, Aptr, D);

                    // B: 16*16 block from K^T, we have K as [BC*D] row-major,
                    // So load B as col_major starting at (col0, k0) with ldb=D
                    wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::precision::bfloat16, wmma::col_major> b_frag;

                    const __nv_bfloat16 *Bptr = Ki + (col * D + kdim);
                    wmma::load_matrix_sync(b_frag, Bptr, D);

                    // Accumulate
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                float *c_tile = warp_scratch_ptr(warp_scratch_base, warp_id);
                wmma::store_matrix_sync(c_tile, c_frag, WN, wmma::mem_row_major);

                __syncwarp();

                if (lane == 0)
                {
#pragma unroll
                    for (int r = 0; r < WM; ++r)
                    {
                        const float *rowp = c_tile + r * WN;

                        float local_max = rowp[0];
#pragma unroll
                        for (int c = 1; c < WN; ++c)
                        {
                            local_max = fmaxf(local_max, rowp[c]);
                        }
                        vmax_rows[r] = fmaxf(vmax_rows[r], local_max * scaling_factor);
                    }
                }

                __syncwarp();
            }

            if (lane == 0)
            {
                for (int r = 0; r < WM; r++)
                {
                    vmax_rows[r] = max(Mi[row + r], vmax_rows[r]);
                }
            }

            __syncwarp();

            for (int r = 0; r < WM; r++)
            {
                for (int dbase = lane; dbase < D; dbase += 32)
                {
                    Oi[(r + row) * D + dbase] *= __expf(vmax_rows[r] - Mi[row + r]);
                }
            }

            // sweep columns in 16 wide tiles
            // Keep per-lane partial row sums to merge into Li after the sweep
            float rowsum_local[WM];
#pragma unroll
            for (int rr = 0; rr < WM; ++rr)
                rowsum_local[rr] = 0.f;

            for (int col = 0; col < BC; col += WN)
            {
                wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);

                // k dimension in 16 chuncks
                for (int kdim = 0; kdim < D; kdim += WK)
                {
                    // A: 16*16 block from Q, row major
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::precision::bfloat16, wmma::row_major> a_frag;

                    const __nv_bfloat16 *Aptr = Qi + (row * D + kdim);
                    wmma::load_matrix_sync(a_frag, Aptr, D);

                    // B: 16*16 block from K^T, we have K as [BC*D] row-major,
                    // So load B as col_major starting at (col0, k0) with ldb=D
                    wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::precision::bfloat16, wmma::col_major> b_frag;
                    const __nv_bfloat16 *Bptr = Ki + (col * D + kdim);
                    wmma::load_matrix_sync(b_frag, Bptr, D);

                    // Accumulate
                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                float *c_tile = warp_scratch_ptr(warp_scratch_base, warp_id);
                wmma::store_matrix_sync(c_tile, c_frag, WN, wmma::mem_row_major);

                __syncwarp();

// Accumulate softmax row sums across all column tiles in this pass
// Use lane-strided iteration to keep work within the warp
#pragma unroll
                for (int t = lane; t < WM * WN; t += 32)
                {
                    int rloc = t / WN; // row within WM tile
                    int cloc = t % WN; // col within WN tile

                    float logit = c_tile[t] * scaling_factor;
                    float p = __expf(logit - vmax_rows[rloc]);

                    rowsum_local[rloc] += p;

                    // Accumulate into output with conflict-free d-strided writes
                    for (int d = lane; d < D; d += 32)
                    {
                        float v = __bfloat162float(Vi[(col + cloc) * D + d]);
                        Oi[(row + rloc) * D + d] += p * v;
                    }
                }

                __syncwarp();
            }
            // After processing all column tiles, compute Li update via warp reduction
#pragma unroll
            for (int rloc = 0; rloc < WM; ++rloc)
            {
                float v = rowsum_local[rloc];
                for (int off = 16; off > 0; off >>= 1)
                {
                    v += __shfl_down_sync(0xffffffff, v, off);
                }
                if (lane == 0)
                {
                    Li[row + rloc] = Li[row + rloc] * __expf(Mi[row + rloc] - vmax_rows[rloc]) + v;
                    Mi[row + rloc] = vmax_rows[rloc];
                }
            }
        }
    }
}