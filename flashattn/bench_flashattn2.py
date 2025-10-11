import os
import math
import time
import csv
from itertools import product

import torch

def benchmark_flashattn2(batch_list=(8,16,32), seqlen_list=(512,1024), heads=8, dim=64,
                         dtype=torch.float16, device='cuda', iters=50, warmup=10,
                         out_csv='bench_flash2.csv'):
    # Try to import flash-attn v2 interface; else fall back to PyTorch SDPA (flash backend)
    use_fa2 = True
    flash_attn_func = None
    try:
        from flash_attn import flash_attn_func  # type: ignore
    except Exception as e:
        print(f"flash-attn import failed ({e}); falling back to torch.nn.functional.scaled_dot_product_attention")
        use_fa2 = False

    results = []

    for B, N in product(batch_list, seqlen_list):
        # Allocate Q,K,V as [B, N, H, D], then reshape to [B*H, N, D] logically inside op
        q = torch.randn(B, N, heads, dim, dtype=dtype, device=device) / math.sqrt(dim)
        k = torch.randn(B, N, heads, dim, dtype=dtype, device=device) / math.sqrt(dim)
        v = torch.randn(B, N, heads, dim, dtype=dtype, device=device) / math.sqrt(dim)

        # Define callables
        if use_fa2:
            def run_once():
                return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False)
            impl_name = 'fa2'
        else:
            import torch.nn.functional as F
            # SDPA expects [B, H, N, D]
            qh = q.permute(0, 2, 1, 3).contiguous()
            kh = k.permute(0, 2, 1, 3).contiguous()
            vh = v.permute(0, 2, 1, 3).contiguous()
            def run_once():
                return F.scaled_dot_product_attention(qh, kh, vh, dropout_p=0.0, is_causal=False)
            impl_name = 'sdpa-flash'

        # Warmup
        for _ in range(warmup):
            o = run_once()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            o = run_once()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters

        flops = 2.0 * B * heads * N * N * dim
        gflops = flops / (ms / 1e3) / 1e9
        results.append({
            'impl': impl_name, 'B': B, 'H': heads, 'N': N, 'D': dim,
            'BR': '', 'BC': '', 'warps': '', 'block_dim': '', 'grid_x': '', 'grid_z': '',
            'smem_bytes': '', 'time_ms': f"{ms:.3f}", 'gflops': f"{gflops:.2f}"
        })
        print(f"[{impl_name}] B={B} N={N} -> {ms:.3f} ms ({gflops:.2f} GF/s)")

    # Write CSV
    fieldnames = ['impl','B','H','N','D','BR','BC','warps','block_dim','grid_x','grid_z','smem_bytes','time_ms','gflops']
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"Wrote results to {out_csv}")

if __name__ == '__main__':
    benchmark_flashattn2()
