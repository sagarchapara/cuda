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

    // Each warp computes 16 rows of O
    for (int row_base = warp_id * WM; row_base < BR; row_base += num_warps * WM)
    {

        // First Pass, compute max of the row
        float m_i[WM]; // max per row
        float l_i[WM]; // sum per row

#pragma unroll
        for (int w = 0; w < WM; ++w)
        {
            m_i[w] = -CUDART_INF_F;
            l_i[w] = 0.0f;
        }

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

            for (int col = 0; col < BC; col += WN)
            {

                // Accumulator tile (WM x WN)
                wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
                wmma::fill_fragment(c_frag, 0.0f);

                for (int k = 0; k < D; k += WK)
                {
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::precision::bfloat16, wmma::row_major> a_frag;
                    wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::precision::bfloat16, wmma::col_major> b_frag;

                    const __nv_float16 *Aptr = &Qi[row_base * D + k];
                    wmma::load_matrix_sync(a_frag, Aptr, D);

                    const __nv__float16 *Bptr = &Ki[col * D + k];
                    wmma::load_matrix_sync(b_frag, Bptr, D);

                    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                float *c_tile = warp_scratch_ptr(warp_scratch_base, warp_id);
                wmma::store_matrix_sync(c_tile, c_frag, WN, wmma::mem_row_major);
                __syncwarp();

// Compute max per row
#pragma unroll
                float m_tile[WM];
                if (lane == 0)
                {
#pragma unroll
                    for (int r = 0; r < WM; ++r)
                    {
                        float row_max = -CUDART_INF_F;
#pragma unroll
                        for (int c = 0; c < WN; ++c)
                        {
                            row_max = fmaxf(row_max, c_tile[r * WN + c] * scaling_factor);
                        }
                        m_tile[r] = row_max;
                    }
                }

#pragma unroll
                for (int r = 0; r < WM; ++r)
                {
                    m_tile[r] = __shfl_sync(0xffffffff, m_tile[r], 0);
                }

#pragma unroll
                for (int r = 0; r < WM; ++r)
                {
                    float new_m = fmaxf(m_tile[r], m_i[r]);
                    float alpha = __expf(m_i[r] - new_m);
                    li[r] *= alpha;
                    mi[r] = new_m;
                }

                // Compute l_i, row sum
                float rowsum_local[WM];
#pragma unroll
                for (int r = 0; r < WM; ++r)
                {
                    rowsum_local[r] = 0.0f;
                }

                for (int t = lane; t < WM * WN; t += 32)
                {
                    int r = t / WN;
                    int c = t % WN;
                    float p = __expf(c_tile[r * WN + c] * scaling_factor - m_i[r]);
                    rowsum_local[r] += p;
                }

                __syncwarp();

#pragma unroll
                for (int r = 0; r < WM; ++r)
                {
                    float sum = row_sum_local[r];
                    // warp reduce
                    for (int offset = 16; offset > 0; offset /= 2)
                    {
                        sum += __shfl_down_sync(0xffffffff, sum, offset);
                    }
                    if (lane == 0)
                    {
                        l_i[r] += sum;
                    }
                }
            }
        }

        for (int dbase = lane; dbase < D; dbase += 32)
        {
            float acc[WM];
#pragma unroll
            for (int w = 0; w < WM; ++w)
            {
                acc[w] = 0.0f;
            }

            for (int j = 0; j < tc; j++)
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

                for (int col = 0; col < BC; col += WN)
                {
                    wmma::fragment<wmma::accumulator, WM, WN, WK, float> c_frag;
                    wmma::fill_fragment(c_frag, 0.0f);

                    for (int k = 0; k < D; k += WK)
                    {
                        wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::precision::bfloat16, wmma::row_major> a_frag;
                        wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::precision::bfloat16, wmma::col_major> b_frag;

                        const __nv_float16 *Aptr = &Qi[row_base * D + k];
                        wmma::load_matrix_sync(a_frag, Aptr, D);

                        const __nv_float16 *Bptr = &Ki[col * D + k];
                        wmma::load_matrix_sync(b_frag, Bptr, D);

                        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }

                    float *c_tile = warp_scratch_ptr(warp_scratch_base, warp_id);
                    wmma::store_matrix_sync(c_tile, c_frag, WN, wmma::mem_row_major);
                    __syncwarp();

                    // Compute output for the rows
                    for (int t = lane; t < WM * WN; t += 32)
                    {
                        int r = t / WN;
                        int c = t % WN;
                        float p = __expf(c_tile[r * WN + c] * scaling_factor - m_i[r]);
                        acc[r] += p * __bfloat162float(Vi[(col + c) * D + dbase]);
                    }
                }
            }

#pragma unroll
            for (int w = 0; w < WM; ++w)
            {
                O[(i + row_base + w) * D + dbase] = acc[w];
            }
        }
    }
}