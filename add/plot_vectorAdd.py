import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

# Usage: python3 plot_vectorAdd.py [vectorAdd_bench.csv]

path = sys.argv[1] if len(sys.argv) > 1 else 'vectorAdd_bench.csv'
rows = []
with open(path, newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        row['blocks'] = int(row['blocks'])
        row['threads'] = int(row['threads'])
        row['avg_ms'] = float(row['avg_ms'])
        row['gbps'] = float(row['gbps'])
        rows.append(row)

# Group by kernel; x-axis can be (blocks*threads) or (threads); we will plot by threads, multiple lines for blocks
kernels = sorted(set(r['kernel'] for r in rows))
blocks_set = sorted(set(r['blocks'] for r in rows))

for k in kernels:
    plt.figure(figsize=(7,4))
    for b in blocks_set:
        xs = []
        ys = []
        for row in rows:
            if row['kernel'] == k and row['blocks'] == b:
                xs.append(row['threads'])
                ys.append(row['gbps'])
        if xs:
            zs = [x for _,x in sorted(zip(xs, ys))]
            xs_sorted = sorted(xs)
            plt.plot(xs_sorted, zs, marker='o', label=f'blocks={b}')
    plt.title(f'{k} bandwidth vs threads per block')
    plt.xlabel('threads per block')
    plt.ylabel('GB/s')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{k}_gbps_vs_threads.png', dpi=150)

# Alternative figure: GB/s vs total threads (blocks*threads)
plt.figure(figsize=(7,4))
for k in kernels:
    xs = []
    ys = []
    for row in rows:
        xs.append(row['blocks']*row['threads'])
        ys.append(row['gbps'])
    plt.scatter(xs, ys, label=k, alpha=0.7)
plt.title('Bandwidth vs total threads (all kernels)')
plt.xlabel('blocks * threads')
plt.ylabel('GB/s')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('all_bandwidth_scatter.png', dpi=150)
print('Saved figures: *_gbps_vs_threads.png and all_bandwidth_scatter.png')
