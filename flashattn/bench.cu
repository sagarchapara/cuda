#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>

#include <cuda_bf16.h>

// Include the kernel implementation directly so we can launch it from here.
#include "forward.cu"

#ifndef CHECK_CUDA
#define CHECK_CUDA(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)
#endif

struct SweepCfg {
    int B;        // batch size
    int H;        // number of heads
    int N;        // seq len
    int D;        // head dim
    int BR;       // tile rows
    int BC;       // tile cols
    int warps;    // warps per block
};
static std::vector<int> parse_list_env(const char *env_name, const std::vector<int> &def) {
    const char *v = getenv(env_name);
    if (!v || !*v) return def;
    std::vector<int> out;
    std::stringstream ss(v);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoi(tok));
    }
    if (out.empty()) return def;
    return out;
}

static float run_kernel_once(const SweepCfg &cfg,
                             const __nv_bfloat16 *Q, const __nv_bfloat16 *K, const __nv_bfloat16 *V,
                             float *O, float *lbuf,
                             float softmax_scale,
                             int iters,
                             std::ostream *dbg = nullptr) {
    // Compute launch dims
    int tr = (cfg.N + cfg.BR - 1) / cfg.BR; // grid.x
    dim3 grid(tr, 1, cfg.B * cfg.H);
    dim3 block(cfg.warps * 32);

    // Shared memory layout in forward.cu (BF16 for Q/K/V tiles, FP32 for others):
    // Qi: BR*D (bf16), Ki: BC*D (bf16), Vi: BC*D (bf16), Mi: BR (f32), Li: BR (f32), Oi: BR*D (f32), MiOld: BR (f32)
    size_t smem_bytes = (size_t)(
        (size_t)cfg.BR * cfg.D * sizeof(__nv_bfloat16) +
        (size_t)cfg.BC * cfg.D * sizeof(__nv_bfloat16) +
        (size_t)cfg.BC * cfg.D * sizeof(__nv_bfloat16) +
        (size_t)cfg.BR * sizeof(float) +
        (size_t)cfg.BR * sizeof(float) +
        (size_t)cfg.BR * cfg.D * sizeof(float) +
        (size_t)cfg.BR * sizeof(float)
    );

    // Warmup
    for (int w = 0; w < 3; ++w) {
        flash_attn_fwd<<<grid, block, smem_bytes>>>(
            cfg.B, cfg.H, cfg.N, cfg.D,
            Q, K, V, O,
            softmax_scale,
            /*mask=*/nullptr,
            /*dropout_prob=*/0.0f,
            /*M=*/cfg.N,
            cfg.BC, cfg.BR,
            lbuf);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Time
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        flash_attn_fwd<<<grid, block, smem_bytes>>>(
            cfg.B, cfg.H, cfg.N, cfg.D,
            Q, K, V, O,
            softmax_scale,
            /*mask=*/nullptr,
            /*dropout_prob=*/0.0f,
            /*M=*/cfg.N,
            cfg.BC, cfg.BR,
            lbuf);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    ms /= iters; // avg ms per iteration
    return ms;
}

static void fill_random(std::vector<__nv_bfloat16> &hb) {
    for (auto &x : hb) {
        float rf = (float)rand() / RAND_MAX - 0.5f; // [-0.5, 0.5]
        x = __float2bfloat16(rf);
    }
}

int main(int argc, char **argv) {
    // Defaults; can be overridden via env vars or args later if desired
    std::vector<int> batch_list = parse_list_env("BENCH_B", {8, 16, 32});
    std::vector<int> seqlen_list = parse_list_env("BENCH_N", {512, 1024});
    std::vector<int> warps_list = parse_list_env("BENCH_WARPS", {4, 8, 16});
    std::vector<int> br_list = parse_list_env("BENCH_BR", {32, 64, 128});
    std::vector<int> bc_list = parse_list_env("BENCH_BC", {32, 64, 128});

    int H = 8;      // number of heads (assumption)
    int D = 64;     // head dimension (assumption)
    int iters = 50; // timing iterations per config
    if (const char *e = getenv("BENCH_ITERS")) {
        iters = std::max(1, atoi(e));
    }

    // Output CSV
    std::string out_csv = "bench.csv";
    if (argc > 1) {
        out_csv = argv[1];
    }
    std::ofstream ofs(out_csv);
    ofs << "impl,B,H,N,D,BR,BC,warps,block_dim,grid_x,grid_z,smem_bytes,time_ms,gflops\n";

    // Seed RNG
    srand(42);

    for (int B : batch_list) {
        for (int N : seqlen_list) {
            // Allocate tensors (contiguous [B*H, N, D])
            size_t elems = (size_t)B * H * N * D;
            std::vector<__nv_bfloat16> hQ(elems), hK(elems), hV(elems);
            fill_random(hQ); fill_random(hK); fill_random(hV);

            __nv_bfloat16 *dQ = nullptr, *dK = nullptr, *dV = nullptr;
            float *dO = nullptr;
            float *dl = nullptr;
            CHECK_CUDA(cudaMalloc(&dQ, elems * sizeof(__nv_bfloat16)));
            CHECK_CUDA(cudaMalloc(&dK, elems * sizeof(__nv_bfloat16)));
            CHECK_CUDA(cudaMalloc(&dV, elems * sizeof(__nv_bfloat16)));
            CHECK_CUDA(cudaMalloc(&dO, elems * sizeof(float)));
            // l buffer sized [B*H, N]
            CHECK_CUDA(cudaMalloc(&dl, (size_t)B * H * N * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dK, hK.data(), elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dV, hV.data(), elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

            float softmax_scale = 1.0f / sqrtf((float)D);

            for (int BR : br_list) {
                if (BR > N) continue; // skip invalid tile
                for (int BC : bc_list) {
                    if (BC > N) continue;
                    for (int warps : warps_list) {
                        SweepCfg cfg{B, H, N, D, BR, BC, warps};

                        int tr = (N + BR - 1) / BR;
                        int grid_z = B * H;
                        int block_dim = warps * 32;
                        // Query device once per (B,N,BR,BC,warps) combo
                        size_t smem_bytes = (size_t)(
                            (size_t)BR * D * sizeof(__nv_bfloat16) +
                            (size_t)BC * D * sizeof(__nv_bfloat16) +
                            (size_t)BC * D * sizeof(__nv_bfloat16) +
                            (size_t)BR * sizeof(float) +
                            (size_t)BR * sizeof(float) +
                            (size_t)BR * D * sizeof(float) +
                            (size_t)BR * sizeof(float)
                        );

                        // Check hardware shared memory per block limit
                        cudaDeviceProp prop{};
                        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
                        // Skip if requested warps exceed max threads per block
                        if (block_dim > prop.maxThreadsPerBlock) {
                            // e.g., 64 warps => 2048 threads would exceed common 1024 threads/block limit
                            continue;
                        }
                        // Allow opt-in to larger smem if available
                        if (smem_bytes > prop.sharedMemPerBlock) {
                            // Try to set attribute for dynamic smem (for cc >= 7.0)
                            cudaFuncAttributes attr{};
                            CHECK_CUDA(cudaFuncGetAttributes(&attr, flash_attn_fwd));
                            // If too big even for max, skip
                            if (smem_bytes > prop.sharedMemPerBlockOptin) {
                                // Skip this config
                                continue;
                            }
                            CHECK_CUDA(cudaFuncSetAttribute(
                                flash_attn_fwd,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (int)smem_bytes));
                        }

                        float ms = run_kernel_once(cfg, dQ, dK, dV, dO, dl, softmax_scale, iters);

                        // Approx flops: 2 * B * H * N * N * D
                        double flops = 2.0 * (double)B * H * N * (double)N * D;
                        double gflops = flops / (ms / 1e3) / 1e9;

                        ofs << "custom" << ','
                            << B << ',' << H << ',' << N << ',' << D << ','
                            << BR << ',' << BC << ',' << warps << ','
                            << block_dim << ',' << tr << ',' << grid_z << ','
                            << smem_bytes << ',' << std::fixed << std::setprecision(3) << ms << ','
                            << std::setprecision(2) << gflops << '\n';
                        ofs.flush();
                        std::cout << "[custom] B=" << B << " N=" << N
                                  << " BR=" << BR << " BC=" << BC
                                  << " warps=" << warps << " -> " << ms << " ms ("
                                  << gflops << " GF/s)" << std::endl;
                    }
                }
            }

            CHECK_CUDA(cudaFree(dQ));
            CHECK_CUDA(cudaFree(dK));
            CHECK_CUDA(cudaFree(dV));
            CHECK_CUDA(cudaFree(dO));
            CHECK_CUDA(cudaFree(dl));
            
        }
    }

    std::cout << "Wrote results to " << out_csv << std::endl;
    return 0;
}
