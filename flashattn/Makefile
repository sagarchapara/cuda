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

.PHONY: all clean
