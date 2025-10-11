# Flash Attention Forward Benchmark

This folder contains:
- `bench.cu`: CUDA benchmark harness for the custom `flash_attn_fwd` kernel (from `forward.cu`).
- `Makefile`: build `bench` binary with NVCC.
- `bench_flashattn2.py`: PyTorch/FlashAttention v2 benchmark to compare against.
- `plot_compare.py`: Plot comparisons and summary images.

## Build the custom CUDA benchmark

Requires CUDA toolkit. From this directory:

```
make -j
```

This produces `./bench`.

Optional env vars to control the sweep:
- `BENCH_B`: batches, e.g. `8,16,32`
- `BENCH_N`: sequence lens, e.g. `512,1024`
- `BENCH_WARPS`: warps per block, e.g. `4,8,16`
- `BENCH_BR`: BR tile sizes, e.g. `32,64,128`
- `BENCH_BC`: BC tile sizes, e.g. `32,64,128`
- `BENCH_ITERS`: iterations per config (default 50)

Run (optionally provide output csv path):

```
./bench bench.csv
```

## Run FlashAttention 2 benchmark

Requires PyTorch with CUDA and `flash-attn` package (v2).

```
python3 bench_flashattn2.py
```

This writes `bench_flash2.csv`.

## Plot results

```
python3 plot_compare.py --custom bench.csv --fa2 bench_flash2.csv --outdir plots
```

Generates per-configuration plots and a `summary_best_vs_fa2.png` in `plots/`.

## Notes
- Kernel assumes H=8 and D=64 in the harness; change inside `bench.cu` if needed.
- FLOPs are approximate (2*B*H*N*N*D) for comparing relative throughput.
- Some (BR,BC) choices may exceed shared memory; those configs are skipped.
