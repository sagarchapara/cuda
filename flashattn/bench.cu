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
                             const float *Q, const float *K, const float *V,
                             float *O, float *lbuf, float *mbuf,
                             float softmax_scale,
                             int iters,
                             std::ostream *dbg = nullptr) {
    // Compute launch dims
    int tr = (cfg.N + cfg.BR - 1) / cfg.BR; // grid.x
    dim3 grid(tr, 1, cfg.B * cfg.H);
    dim3 block(cfg.warps * 32);

    size_t smem_floats = (size_t)(2 * cfg.BR * cfg.D   // Qi + Oi
                                  + 2 * cfg.BC * cfg.D // Ki + Vi
                                  + 3 * cfg.BR);       // Mi + Li + MiOld
    size_t smem_bytes = smem_floats * sizeof(float);

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
            lbuf, mbuf);
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
            lbuf, mbuf);
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

static void fill_random(std::vector<float> &h) {
    for (auto &x : h) {
        x = (float)rand() / RAND_MAX - 0.5f; // [-0.5, 0.5]
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
            std::vector<float> hQ(elems), hK(elems), hV(elems);
            fill_random(hQ); fill_random(hK); fill_random(hV);

            float *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
            float *dl = nullptr, *dm = nullptr;
            CHECK_CUDA(cudaMalloc(&dQ, elems * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dK, elems * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dV, elems * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dO, elems * sizeof(float)));
            // l, m buffers sized [B*H, N]
            CHECK_CUDA(cudaMalloc(&dl, (size_t)B * H * N * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&dm, (size_t)B * H * N * sizeof(float)));
            CHECK_CUDA(cudaMemcpy(dQ, hQ.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dK, hK.data(), elems * sizeof(float), cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(dV, hV.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

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
                        size_t smem_floats = (size_t)(2 * BR * D + 2 * BC * D + 3 * BR);
                        size_t smem_bytes = smem_floats * sizeof(float);

                        // Check hardware shared memory per block limit
                        cudaDeviceProp prop{};
                        CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
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

                        float ms = run_kernel_once(cfg, dQ, dK, dV, dO, dl, dm, softmax_scale, iters);

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
            CHECK_CUDA(cudaFree(dm));
        }
    }

    std::cout << "Wrote results to " << out_csv << std::endl;
    return 0;
}
