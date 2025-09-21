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

// This is same as global memory coalescing.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float value = 0;
        for (int e = 0; e < K; ++e)
            value += A[row * K + e] * B[e * N + col];

        // C = alpha * A * B + beta * C
        C[row * N + col] = alpha * value + beta * C[row * N + col];
    }
}

__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const int BLOCK_SIZE = 32;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    float tmp = 0;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE)
    {

        // Read one row and column per thread
        As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[row * K + (bkIdx + threadIdx.x)];
        Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(bkIdx + threadIdx.y) * N + col];

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e)
            tmp += As[threadIdx.y * BLOCK_SIZE + e] * Bs[e * BLOCK_SIZE + threadIdx.x];

        __syncthreads();
    }

    C[row * N + col] = alpha * tmp + beta * C[row * N + col];
}

__global__ void sgemm_1d_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;

    // We are calculating BM x BN output matrix C, using BM * BN / TM threads
    // We are using warp colescing, so each thread will calculate TM elements in a column
    // Going for 1D threads implementation.

    const int threadsPerBlock = (BM * BN) / TM;

    assert(blockDim.x == threadsPerBlock);

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    float threadResults[TM] = {0.0};

    const int threadRow = blockRow + (threadIdx.x / BN) * TM;
    const int threadCol = blockCol + (threadIdx.x % BN);

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK)
    {
        __shared__ float As[BM * BK]; // 64 * 8
        __shared__ float Bs[BK * BN]; // 8 * 64

        const int aRow = blockRow + (threadIdx.x / BK);
        const int aCol = bkIdx + (threadIdx.x % BK);

        const int bRow = bkIdx + (threadIdx.x / BN);
        const int bCol = blockCol + (threadIdx.x % BN);

        // Load As and Bs
        As[threadIdx.x] = A[aRow * K + aCol];
        Bs[threadIdx.x] = B[bRow * N + bCol];

        __syncthreads();

        for (int k = 0; k < BK; k++)
        {
            float btmp = Bs[k * BN + threadCol - blockCol];
            for (int tm = 0; tm < TM; tm++)
            {
                threadResults[tm] += As[(tm + threadRow - blockRow) * BK + k] * btmp;
            }
        }

        __syncthreads();
    }

    for (int tm = 0; tm < TM; tm++)
    {
        C[(tm + threadRow) * N + threadCol] = alpha * threadResults[tm] + beta * C[(tm + threadRow) * N + threadCol];
    }
}

__global__ void sgemm_2d_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int numThreadsPerBlock = (BM * BN) / (TM * TN);

    assert(blockDim.x == numThreadsPerBlock);

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int threadRow = (threadIdx.x / (BN / TN)) * TM;
    const int threadCol = (threadIdx.x % (BN / TN)) * TN;

    float threadResults[TM * TN] = {0.0};

    // register cache for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        const int strideA = numThreadsPerBlock;
        const int strideB = numThreadsPerBlock;

        for (uint i = 0; i < TM; i++)
        {
            const int rowA = (threadIdx.x + strideA * i) / BK;
            const int colA = (threadIdx.x + strideA * i) % BK;

            As[rowA * BK + colA] = A[(blockRow + rowA) * K + (bkIdx + colA)];
        }

        for (uint i = 0; i < TN; i++)
        {
            const int rowB = (threadIdx.x + strideB * i) / BN;
            const int colB = (threadIdx.x + strideB * i) % BN;

            Bs[rowB * BN + colB] = B[(bkIdx + rowB) * N + (blockCol + colB)];
        }

        __syncthreads();

        for (uint k = 0; k < BK; k++)
        {
            for (uint row = 0; row < TM; ++row)
            {
                regM[row] = As[(threadRow + row) * BK + k];
            }

            for (uint col = 0; col < TN; ++col)
            {
                regN[col] = Bs[k * BN + threadCol + col];
            }

            for (uint row = 0; row < TM; ++row)
            {
                for (uint col = 0; col < TN; ++col)
                {
                    threadResults[row * TN + col] += regM[row] * regN[col];
                }
            }
        }

        __syncthreads();
    }

    for (uint row = 0; row < TM; ++row)
    {
        for (uint col = 0; col < TN; ++col)
        {
            C[(blockRow + threadRow + row) * N + (blockCol + threadCol + col)] = alpha * threadResults[row * TN + col] + beta * C[(blockRow + threadRow + row) * N + (blockCol + threadCol + col)];
        }
    }
}

__global__ void sgemm_2d_tiling_vectorized(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C)
{

    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const int numThreadsPerBlock = (BM * BN) / (TM * TN);

    assert(blockDim.x == numThreadsPerBlock);

    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    const int threadRow = (threadIdx.x / (BN / TN)) * TM;
    const int threadCol = (threadIdx.x % (BN / TN)) * TN;

    float threadResults[TM * TN] = {0.0};

    // register cache for As and Bs
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK)
    {

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        const int rowA = (threadIdx.x / (BK / 4));
        const int colA = 4 * (threadIdx.x % (BK / 4));

        const int rowB = (threadIdx.x / (BN / 4));
        const int colB = 4 * (threadIdx.x % (BN / 4));

        // Load As and Bs
        float4 tmp = reinterpret_cast<const float4 *>(&A[(blockRow + rowA) * K + (bkIdx + colA)])[0];

        As[(colA + 0) * BM + rowA] = tmp.x;
        As[(colA + 1) * BM + rowA] = tmp.y;
        As[(colA + 2) * BM + rowA] = tmp.z;
        As[(colA + 3) * BM + rowA] = tmp.w;

        reinterpret_cast<float4 *>(&Bs[rowB * BN + colB])[0] = reinterpret_cast<const float4 *>(&B[(bkIdx + rowB) * N + (blockCol + colB)])[0];

        __syncthreads();

        for (uint k = 0; k < BK; k++)
        {
            for (uint row = 0; row < TM; ++row)
            {
                regM[row] = As[k * BM + threadRow + row]; // Sequential access, so 128B load.
            }

            for (uint col = 0; col < TN; ++col)
            {
                regN[col] = Bs[k * BN + threadCol + col];
            }

            for (uint row = 0; row < TM; ++row)
            {
                for (uint col = 0; col < TN; ++col)
                {
                    threadResults[row * TN + col] += regM[row] * regN[col];
                }
            }
        }

        __syncthreads();
    }

    for (uint row = 0; row < TM; ++row)
    {
        for (uint col = 0; col < TN; col += 4)
        {
            float4 tmp = reinterpret_cast<float4 *>(&C[(blockRow + threadRow + row) * N + (blockCol + threadCol + col)])[0];

            tmp.x = alpha * threadResults[row * TN + col] + beta * tmp.x;
            tmp.y = alpha * threadResults[row * TN + col + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[row * TN + col + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[row * TN + col + 3] + beta * tmp.w;

            reinterpret_cast<float4 *>(&C[(blockRow + threadRow + row) * N + (blockCol + threadCol + col)])[0] = tmp;
        }
    }
}
