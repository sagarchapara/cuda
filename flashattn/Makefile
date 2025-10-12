NVCC ?= nvcc
CUDA_ARCH ?= -gencode arch=compute_80,code=sm_80 \
             -gencode arch=compute_86,code=sm_86 \
             -gencode arch=compute_89,code=sm_89 \
             -gencode arch=compute_90,code=sm_90

CXXFLAGS ?= -O3 -std=c++17 -lineinfo
LDFLAGS  ?=

TARGET = bench
SRCS = bench.cu

all: $(TARGET)

$(TARGET): $(SRCS) forward.cu
	$(NVCC) $(CXXFLAGS) $(CUDA_ARCH) -Xptxas -O3 --use_fast_math -o $@ $(SRCS)

clean:
	rm -f $(TARGET)

.PHONY: profile-ncu profile-nsys plot

# ------------------------------------------------------------
# Profiling helpers (single-config runs)
# Override on the command line, e.g.:
#   make profile-ncu B=8 N=1024 BR=64 BC=64 WARPS=8
# ------------------------------------------------------------
B ?= 8
N ?= 512
H ?= 8
D ?= 64
BR ?= 64
BC ?= 64
WARPS ?= 8

# Nsight Compute flags
NCU ?= ncu
NCU_FLAGS ?= --set full --target-processes application-only \
             --kernel-name 'flash_attn_fwd' \
             --launch-skip 3 --launch-count 1
# Nsight Systems flags
NSYS ?= nsys
NSYS_FLAGS ?= profile --trace=cuda,osrt --sample=none -o flashattn_nsys

profile-ncu: $(TARGET)
	@echo "[NCU] Profiling single config: B=$(B) N=$(N) BR=$(BR) BC=$(BC) WARPS=$(WARPS)"
	BENCH_B=$(B) BENCH_N=$(N) BENCH_BR=$(BR) BENCH_BC=$(BC) BENCH_WARPS=$(WARPS) BENCH_ITERS=1 \
		$(NCU) $(NCU_FLAGS) --export flashattn_ncu \
		./$(TARGET) bench.csv
	@echo "NCU export written (flashattn_ncu.*). Open .ncu-rep in Nsight Compute GUI if desired."

profile-nsys: $(TARGET)
	@echo "[NSYS] Profiling single config: B=$(B) N=$(N) BR=$(BR) BC=$(BC) WARPS=$(WARPS)"
	BENCH_B=$(B) BENCH_N=$(N) BENCH_BR=$(BR) BENCH_BC=$(BC) BENCH_WARPS=$(WARPS) BENCH_ITERS=1 \
		$(NSYS) $(NSYS_FLAGS) ./$(TARGET) bench.csv
	@echo "NSYS report: flashattn_nsys.qdrep / .nsys-rep"

# Plot with Python venv
plot:
	. ../.venv/bin/activate && python3 plot_compare.py --custom bench.csv --fa2 bench_flash2.csv --outdir plots
	@echo "Plots saved under flashattn/plots" 

.PHONY: all clean
