#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

#include <cuda_runtime.h>


__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vectorAddStrided(const float *A, const float *B, float *C, int numElements){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < numElements; i += stride) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vectorAddVectorized(const float *A, const float *B, float *C, int numElements){
    // Grid-stride over float4 chunks, then handle tail for leftover elements
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int n4 = numElements / 4;
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
    float4* C4 = reinterpret_cast<float4*>(C);

    for (int i4 = tid; i4 < n4; i4 += stride) {
        float4 a = A4[i4];
        float4 b = B4[i4];
        C4[i4] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }

    int base = n4 * 4;
    for (int i = base + tid; i < numElements; i += stride) {
        C[i] = A[i] + B[i];
    }
}

// Simple helper for CUDA error checking in bench mode
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s (error %s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

static float runKernelTimed(int which, int blocks, int threads, int repeats,
                            const float* d_A, const float* d_B, float* d_C, int n) {
    // which: 0=naive, 1=strided, 2=vectorized
    cudaEvent_t s, e;
    checkCuda(cudaEventCreate(&s), "event create s");
    checkCuda(cudaEventCreate(&e), "event create e");

    // Warm-up
    switch (which) {
        case 0: vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, n); break;
        case 1: vectorAddStrided<<<blocks, threads>>>(d_A, d_B, d_C, n); break;
        case 2: vectorAddVectorized<<<blocks, threads>>>(d_A, d_B, d_C, n); break;
    }
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    checkCuda(cudaEventRecord(s), "event record start");
    for (int r = 0; r < repeats; ++r) {
        switch (which) {
            case 0: vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, n); break;
            case 1: vectorAddStrided<<<blocks, threads>>>(d_A, d_B, d_C, n); break;
            case 2: vectorAddVectorized<<<blocks, threads>>>(d_A, d_B, d_C, n); break;
        }
    }
    checkCuda(cudaEventRecord(e), "event record end");
    checkCuda(cudaEventSynchronize(e), "event sync end");
    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, s, e), "elapsed time");

    checkCuda(cudaEventDestroy(s), "event destroy s");
    checkCuda(cudaEventDestroy(e), "event destroy e");
    return ms / repeats; // average per launch
}

int main(void){
    cudaError_t err = cudaSuccess;

    int numElements = 1<<20; // 1M elements for more stable timing
    size_t size = numElements * sizeof(float);

    printf("[Vector addition of %d elements]\n", numElements);

    // Measure overall wall time
    auto t0 = std::chrono::high_resolution_clock::now();

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; i++) {
        h_A[i] = rand() / (float)(RAND_MAX);
        h_B[i] = rand() / (float)(RAND_MAX);
    }

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    // Report device memory before allocations
    size_t free0 = 0, total0 = 0;
    err = cudaMemGetInfo(&free0, &total0);
    if (err == cudaSuccess) {
        printf("Device memory before alloc: free=%zu MB, total=%zu MB\n", free0 / (1024 * 1024), total0 / (1024 * 1024));
    }

    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Create CUDA events for timing
    cudaEvent_t evStart, evStop, evKStart, evKStop, evD2HStart, evD2HStop;
    cudaEventCreate(&evStart);
    cudaEventCreate(&evStop);
    cudaEventCreate(&evKStart);
    cudaEventCreate(&evKStop);
    cudaEventCreate(&evD2HStart);
    cudaEventCreate(&evD2HStop);

    // Time H2D transfers
    cudaEventRecord(evStart, 0);
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(evStop, 0);
    cudaEventSynchronize(evStop);
    float msH2D = 0.0f;
    cudaEventElapsedTime(&msH2D, evStart, evStop);

    // Benchmark sweep mode via env: set VECADD_BENCH=1 to run sweeps and CSV output
    const char* bench = getenv("VECADD_BENCH");
    float msKernel = 0.0f;
    if (bench && bench[0] == '1') {
        std::vector<int> threadsList = {64, 128, 256, 512};
        // choose a few grid sizes relative to N to vary stride
        std::vector<int> blocksList = { (numElements+255)/256/4, (numElements+255)/256/2, (numElements+255)/256, 2*((numElements+255)/256) };
        int repeats = 50;

        std::ofstream csv("vectorAdd_bench.csv");
        csv << "kernel,blocks,threads,avg_ms,gbps\n";
        double gb = (double)size / 1e9;
        for (int which = 0; which < 3; ++which) {
            for (int threads : threadsList) {
                for (int blocks : blocksList) {
                    if (threads <= 0 || blocks <= 0) continue;
                    float ms = runKernelTimed(which, blocks, threads, repeats, d_A, d_B, d_C, numElements);
                    double gbps = gb / (ms / 1e3);
                    const char* kname = (which==0?"naive":(which==1?"strided":"vectorized"));
                    printf("%s blocks=%d threads=%d avg_ms=%.3f bw=%.2f GB/s\n", kname, blocks, threads, ms, gbps);
                    csv << kname << "," << blocks << "," << threads << "," << std::fixed << std::setprecision(3) << ms << "," << std::setprecision(2) << gbps << "\n";
                }
            }
        }
        csv.close();
        msKernel = -1.0f; // not meaningful in sweep path
    } else {
        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1)/threadsPerBlock;
        printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
        cudaEventRecord(evKStart, 0);
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        cudaEventRecord(evKStop, 0);
        cudaEventSynchronize(evKStop);

        err = cudaGetLastError();
        if (err != cudaSuccess){
            fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaEventElapsedTime(&msKernel, evKStart, evKStop);
    }

    printf("Copy output data from the CUDA device to the host memory\n");
    // Time D2H transfer
    cudaEventRecord(evD2HStart, 0);
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaEventRecord(evD2HStop, 0);
    cudaEventSynchronize(evD2HStop);
    float msD2H = 0.0f;
    cudaEventElapsedTime(&msD2H, evD2HStart, evD2HStop);

    // Print timing summary
    double gb = (double)size / 1e9; // GB
    printf("Timing (ms): H2D=%.3f (%.2f GB/s), Kernel=%s, D2H=%.3f (%.2f GB/s)\n",
           msH2D, gb / (msH2D / 1e3), (msKernel>=0? (std::to_string(msKernel)+" ms").c_str():"(sweep)"), msD2H, gb / (msD2H / 1e3));

    // Report device memory after allocations (before free)
    size_t free1 = 0, total1 = 0;
    if (cudaMemGetInfo(&free1, &total1) == cudaSuccess) {
        printf("Device memory after alloc:  free=%zu MB, total=%zu MB (delta=-%zu MB)\n",
               free1 / (1024 * 1024), total1 / (1024 * 1024),
               (free0 - free1) / (1024 * 1024));
    }

    err = cudaFree(d_A);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_C);

    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Report device memory after free
    size_t free2 = 0, total2 = 0;
    if (cudaMemGetInfo(&free2, &total2) == cudaSuccess) {
        printf("Device memory after free:  free=%zu MB, total=%zu MB (recovered=+%zu MB)\n",
               free2 / (1024 * 1024), total2 / (1024 * 1024),
               (free2 - free1) / (1024 * 1024));
    }

    free(h_A);
    free(h_B);
    free(h_C);
    
    // Overall wall time and host memory usage (RSS)
    auto t1 = std::chrono::high_resolution_clock::now();
    double wallMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        // ru_maxrss is kilobytes on Linux
        printf("Overall wall time: %.3f ms, Peak RSS: %ld MB\n", wallMs, ru.ru_maxrss / 1024);
    } else {
        printf("Overall wall time: %.3f ms\n", wallMs);
    }

    return 0;
}