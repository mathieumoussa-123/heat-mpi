#!/usr/bin/env python3
"""Fit t_oneway = t_s + m * t_w to one ping-pong CSV.
Latency t_s is estimated from the median of messages <= 64 B; with t_s fixed,
t_w (seconds/byte) is least-squares fitted using messages >= 64 KiB.
This separation reduces bias from different small/large-message protocols.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", type=Path)
args = parser.parse_args()
data = np.genfromtxt(args.csv_file, delimiter=",", names=True)
m = np.asarray(data["bytes"], dtype=float)
t = np.asarray(data["mean_oneway_s"], dtype=float)
small = m <= 64
large = m >= (64 * 1024)
if not (np.any(small) and np.any(large)):
    raise SystemExit("Need both small (<=64 B) and large (>=64 KiB) messages")
t_s = float(np.median(t[small]))
t_w = float(np.dot(m[large], t[large] - t_s) / np.dot(m[large], m[large]))
if t_w <= 0:
    raise SystemExit("Nonpositive fitted time-per-byte; inspect measurements")
bandwidth_mib_s = 1.0 / t_w / (1024.0 ** 2)
print(f"file={args.csv_file}")
print(f"t_s_us={t_s * 1e6:.6f}")
print(f"t_w_s_per_byte={t_w:.12e}")
print(f"bandwidth_MiB_per_s={bandwidth_mib_s:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.loglog(m, t, "o-", label="Measured one-way time")
ax.loglog(m, t_s + m * t_w, "--", label="Latency + bandwidth model")
ax.set_xlabel("Message size (bytes)")
ax.set_ylabel("One-way time (seconds)")
ax.set_title(f"MPI ping-pong: {args.csv_file.stem}")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
plot_path = args.csv_file.with_suffix(".png")
fig.savefig(plot_path, dpi=160)
print(f"plot={plot_path}")
