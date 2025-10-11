import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, 'r') as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows


def filter_rows(rows, **kwargs):
    out = []
    for row in rows:
        ok = True
        for k, v in kwargs.items():
            if v is None:
                continue
            # Cast ints when possible
            if k in row and row[k] != '':
                rv = int(row[k]) if row[k].isdigit() else row[k]
            else:
                rv = row.get(k, '')
            if rv != v:
                ok = False
                break
        if ok:
            out.append(row)
    return out


def plot_vs_fa2(custom_rows, fa2_rows, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # Plot 1: GFLOPs vs warps for each (B,N) and each (BR,BC)
    # We'll generate subplots per (B,N,BR,BC)
    key_groups = defaultdict(list)
    for r in custom_rows:
        key = (int(r['B']), int(r['N']), int(r['BR']), int(r['BC']))
        key_groups[key].append(r)

    for (B, N, BR, BC), rows in key_groups.items():
        rows = sorted(rows, key=lambda x: int(x['warps']))
        x = [int(r['warps']) for r in rows]
        y = [float(r['gflops']) for r in rows]

        # Find matching FA2 row for same (B,N)
        fa2 = [r for r in fa2_rows if int(r['B']) == B and int(r['N']) == N]
        fa2_gflops = float(fa2[0]['gflops']) if fa2 else None

        plt.figure(figsize=(6,4))
        plt.plot(x, y, marker='o', label=f'custom BR={BR} BC={BC}')
        if fa2_gflops is not None:
            plt.axhline(fa2_gflops, color='r', linestyle='--', label='FlashAttention-2')
        plt.xlabel('Warps per block')
        plt.ylabel('GFLOPs/s (approx)')
        plt.title(f'B={B}, N={N}, D=64, H=8')
        plt.legend()
        plt.grid(True, alpha=0.3)
        fname = outdir / f"compare_B{B}_N{N}_BR{BR}_BC{BC}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    # Plot 2: For each (B,N) choose the best custom config vs FA2 as bar
    best_by_bn = {}
    for r in custom_rows:
        key = (int(r['B']), int(r['N']))
        g = float(r['gflops'])
        if key not in best_by_bn or g > best_by_bn[key]:
            best_by_bn[key] = g

    bn_keys = sorted(best_by_bn.keys())
    xlab = [f"B{b}-N{n}" for (b,n) in bn_keys]
    custom_best = [best_by_bn[k] for k in bn_keys]
    fa2_map = {(int(r['B']), int(r['N'])): float(r['gflops']) for r in fa2_rows}
    fa2_vals = [fa2_map.get(k, 0.0) for k in bn_keys]

    import numpy as np
    x = np.arange(len(bn_keys))
    w = 0.35
    plt.figure(figsize=(8,4))
    plt.bar(x - w/2, custom_best, width=w, label='Custom (best)')
    plt.bar(x + w/2, fa2_vals, width=w, label='FlashAttention-2')
    plt.xticks(x, xlab, rotation=0)
    plt.ylabel('GFLOPs/s (approx)')
    plt.title('Best Custom vs FA2 by (B,N)')
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    out = outdir / 'summary_best_vs_fa2.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--custom', default='bench.csv', help='CSV from custom CUDA bench')
    ap.add_argument('--fa2', default='bench_flash2.csv', help='CSV from flash-attn2 bench')
    ap.add_argument('--outdir', default=None, help='Output directory for plots (default: flashattn/plots)')
    args = ap.parse_args()

    script_dir = Path(__file__).parent
    outdir = Path(args.outdir) if args.outdir else (script_dir / 'plots')

    custom_rows = read_csv(args.custom)
    fa2_rows = read_csv(args.fa2)

    plot_vs_fa2(custom_rows, fa2_rows, outdir)


if __name__ == '__main__':
    main()
