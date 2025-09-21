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

__global__ void sum_naive(int N, const float *x, float *y)
{
    const int blockSize = 256;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[blockSize];

    sdata[threadIdx.x] = (idx < N) ? x[idx] : 0.0f;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2)
    {
        int tIdx = 2 * s * threadIdx.x;

        if (tIdx < blockDim.x)
        {
            sdata[tIdx] += sdata[tIdx + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        y[blockIdx.x] = sdata[threadIdx.x];
    }
}

__global__ void sum_serial(int N, const float *x, float *y)
{
    const int blockSize = 256;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[blockSize];

    sdata[threadIdx.x] = (idx < N) ? x[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        y[blockIdx.x] = sdata[threadIdx.x];
    }
}

__global__ void sum_unroll(int N, const float *x, float *y)
{
    const int blockSize = 256;

    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    assert(blockSize == 2*blockDim);

    __shared__ float sdata[blockSize];

    sdata[threadIdx.x] = (idx < N) ? x[idx] : 0.0f;
    if (idx + blockDim.x < N)
    {
        sdata[threadIdx.x] += x[idx + blockDim.x];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        y[blockIdx.x] = sdata[threadIdx.x];
    }
}

__global__ void sum_warp_reduce(int N, const float *x, float *y)
{
    const int blockSize = 256;

    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    assert(blockSize == 2*blockDim);

    __shared__ float sdata[blockSize];

    sdata[threadIdx.x] = (idx < N) ? x[idx] : 0.0f;
    if (idx + blockDim.x < N)
    {
        sdata[threadIdx.x] += x[idx + blockDim.x];
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32)
    {
        // the speed improvement is due to not using __syncthreads, as all these are in single warp
        volatile float *vsmem = sdata;
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 32];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 16];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 8];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 4];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 2];
        vsmem[threadIdx.x] += vsmem[threadIdx.x + 1];
    }

    if (threadIdx.x == 0)
    {
        y[blockIdx.x] = sdata[threadIdx.x];
    }
}
