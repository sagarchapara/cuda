#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

#include <cuda_runtime.h>
#include <assert.h>

template <bool kCausal, bool kSliding>
struct MaskPolicy
{
    int win_left; // only used if kSliding
    int win_right;
    int seq_len_q;
    int seq_len_k;

    __device__ __forceinline__ bool operator()(int q, int k) const
    {
        if (q >= seq_len_q || k >= seq_len_k)
            return true;
        bool m = false;
        if constexpr (kCausal)
        {
            m |= (k > q);
        }
        if constexpr (kSliding)
        {
            if (win_left >= 0)
                m |= (k < q - win_left);
            if (win_right >= 0)
                m |= (k > q + win_right);
        }
        return m;
    }
    __device__ __forceinline__ float addend(int q, int k) const
    {
        return (*this)(q, k) ? -INFINITY : 0.f;
    }
};

__global__ void flash_attn_fwd(int B, int H, int D, const float *Q, const float *K, const float *V, float *O, float scaling_factor, const bool *mask, float dropout_prob, const int M, int BC, int BR, float *l, float *m)
{

    int tc = (n + bc - 1) / bc;
    int tr = (n + br - 1) / br;

    __shared__ float Qi[BR * D];
    __shared__ float Ki[BC * D];
    __shared__ float Vi[BC * D];
    __shared__ float S[BR * BC];
    __shared__ float Mi[BR];
    __shared__ float Li[BR];
    __shared__ float Oi[BR * D];
    __shared__ float Pi[BR * BC];

    const int batch_offset = gridDim.x * blockIdx.x * H * D;

    for (int i = 0; i < tr; i++)
    {
        // load the Qi to shared memory
        for (int j = 0; j * blockDim.x + threadIdx.x < (BR * D); j++)
        {
            int index = j * blockDim.x + threadIdx.x;
            if (index < BR * D)
            {
                int row = index / D;
                int col = index - row * D; // equivalent to index % D but slightly cheaper
                Qi[row * D + col] = Q[batch_offset + index + i * BR * D];
                Mip[row] = -CUDART_INF_F;
                Lip[row] = 0.0f;
                Oi[row * D + col] = 0.0f;
            }
        }

        __syncthreads();

        for (int j = 0; j < tc; j++)
        {
            // load the Ki and Vi to shared memory
            for (int k = 0; k * blockDim.x + threadIdx.x < (BC * D); k++)
            {
                int index = k * blockDim.x + threadIdx.x;
                if (index < BC * D)
                {
                    int row = index / D;
                    int col = index - row * D; // equivalent to index % D but slightly cheaper
                    Ki[row * D + col] = K[batch_offset + index + j * BC * D];
                    Vi[row * D + col] = V[batch_offset + index + j * BC * D];
                }
            }

            __syncthreads();

            // compute matrix multiplication of S
            for (int k = 0; (threadIdx.x + k * blockDim.x) < (BR * BC); k++)
            {
                int index = threadIdx.x + k * blockDim.x;
                if (index < BR * BC)
                {
                    int row = index / BC;
                    int col = index - row * BC; // equivalent to index % BC but slightly cheaper
                    float sum = 0.0;
                    for (int e = 0; e < D; e++)
                    {
                        sum += Qi[row * D + e] * Ki[col * D + e];
                    }
                    S[row * BC + col] = sum * scaling_factor;
                }
            }

            __syncthreads();

            // Warp-per-row reduction: each warp scans all columns of its assigned rows in 32-wide tiles.
            // Eliminates the multi-stride shared-memory halving + barriers.
            // Assumptions:
            //  - blockDim.x is a multiple of 32.
            //  - BR can be >= num_warps; we iterate rows in a strided fashion per warp.
            //  - S holds BR x BC scores (already computed). We compute Mi[row] = max over columns.
            int lane = threadIdx.x & 31;
            int warp_id = threadIdx.x >> 5;
            int num_warps = blockDim.x >> 5;

            __shared__ float MiOld[BR];

            for (int row = warp_id; row < BR; row += num_warps)
            {
                float vmax = -CUDART_INF_F;
                // Iterate over column tiles of width 32.
                for (int base = 0; base < BC; base += 32)
                {
                    int col = base + lane;
                    float val = (col < BC) ? S[row * BC + col] : -CUDART_INF_F;
                    vmax = fmaxf(vmax, val);
                }
// In-warp reduce vmax.
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                {
                    float other = __shfl_down_sync(0xffffffff, vmax, offset);
                    vmax = fmaxf(vmax, other);
                }
                if (lane == 0)
                    MiOld[row] = Mi[row];
                Mi[row] = fmaxf(vmax, Mi[row]);
            }
            __syncthreads();

            // Compute Li[row] = sum_j exp(S[row,j] - Mi[row])
            for (int row = warp_id; row < BR; row += num_warps)
            {
                float lsum = 0.0f;
                float mval = Mi[row];
                for (int base = 0; base < BC; base += 32)
                {
                    int col = base + lane;
                    float val = (col < BC) ? S[row * BC + col] : -CUDART_INF_F;
                    float p = __expf(val - mval);
                    Pi[row * BC + col] = p; // store the probability for later use
                    lsum += p;
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

            // compute Oi = diag(exp(MOld - M)) @ Oi + Pi @ Vi
            for (int k = 0; (threadIdx.x + k * blockDim.x) < (BR * D); k++)
            {
                int index = threadIdx.x + k * blockDim.x;
                int row = index / D;
                int col = index - row * D; // equivalent to index % D but slightly cheaper

                Oi[row * D + col] = Oi[row * D + col] * __expf(-MiOld[row] + Mi[row]);

                for (int e = 0; e < BC; e++)
                {
                    float p = Pi[row * BC + e];
                    float v = Vi[e * D + col];
                    Oi[row * D + col] += p * v;
                }
            }

            __syncthreads();
        }

        // write the Oi to global memory
        for (int j = 0; j * blockDim.x + threadIdx.x < (BR * D); j++)
        {
            int index = j * blockDim.x + threadIdx.x;
            int row = index / D;
            int col = index - row * D; // equivalent to index % D but slightly cheaper
            O[batch_offset + index + i * BR * D] = Oi[row * D + col] / Li[row];

            if (col == 0)
            {
                L[batch_offset / D + i * BR + row] = Li[row];
            }
        }
    }
}