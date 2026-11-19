#!/usr/bin/env python3
"""Fit MPI ping-pong results and compare experiments on shared axes.

Compatible with the original one-file invocation:
    python3 scripts/fit_pingpong.py results/ping_heat-mathieu_heat-naji.csv

For this project's actual experiments (SHM on each VM, TCP over Tailscale
between VMs; NO same-VM TCP benchmark):
    python3 scripts/fit_pingpong.py --compare-dir results
    python3 scripts/fit_pingpong.py --compare-dir results --individual

The comparison uses CSV measurements, not previously rendered PNGs.
The fit is unchanged from the starter: t_s is the median one-way time for
messages <=64 B and t_w is a least-squares slope (fixed intercept t_s)
using messages >=64 KiB: t_oneway = t_s + m*t_w.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SMALL_LIMIT = 64
LARGE_LIMIT = 64 * 1024
EXPECTED_SIZES = 22  # 8, 16, ..., 16 MiB


@dataclass
class Experiment:
    path: Path
    label: str
    group: int  # 0: same-VM SHM; 1: inter-VM Tailscale
    sizes: np.ndarray
    times: np.ndarray
    t_s: float
    t_w: float

    @property
    def bandwidth_mib_s(self):
        return 1.0 / (self.t_w * 1024.0 ** 2)


def display_host(host):
    """Make 'heat-mathieu' readable in plot legends."""
    return re.sub(r"^heat-", "", host, flags=re.IGNORECASE).replace("-", " ").title()


def classify_filename(path):
    """Return (human label, group), ignoring non-experiment filenames.

    Supported names:
        ping_same_heat-mathieu_shm.csv
        ping_heat-mathieu_heat-naji.csv
    """
    name = path.name
    match = re.fullmatch(r"ping_same_(.+)_shm\.csv", name, flags=re.IGNORECASE)
    if match:
        return f"{display_host(match.group(1))} (same VM, SHM)", 0

    # Hostnames for this cluster contain hyphens but not underscores.
    match = re.fullmatch(r"ping_([^_]+)_([^_]+)\.csv", name, flags=re.IGNORECASE)
    if match and not any(part.lower() == "same" for part in match.groups()):
        return f"{display_host(match.group(1))} ↔ {display_host(match.group(2))}", 1

    return None


def load_experiment(path, *, require_known_name=False):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing CSV file: {path}")
    kind = classify_filename(path)
    if require_known_name and kind is None:
        raise ValueError(f"Unrecognized experiment CSV filename: {path.name}")
    label, group = kind if kind is not None else (path.stem, 1)

    rows = []
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"bytes", "mean_oneway_s"}.issubset(reader.fieldnames):
            raise ValueError(f"{path}: expected 'bytes' and 'mean_oneway_s' columns")
        for line, row in enumerate(reader, start=2):
            try:
                size = int(row["bytes"])
                time = float(row["mean_oneway_s"])
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path}:{line}: invalid/incomplete measurement: {row}") from exc
            if size <= 0 or not np.isfinite(time) or time <= 0:
                raise ValueError(f"{path}:{line}: bytes and mean_oneway_s must be positive and finite")
            rows.append((size, time))

    if not rows:
        raise ValueError(f"{path}: no measurements found")
    if len({size for size, _ in rows}) != len(rows):
        raise ValueError(f"{path}: duplicate message sizes detected; fix CSV before fitting")
    rows.sort(key=lambda item: item[0])
    sizes = np.asarray([size for size, _ in rows], dtype=float)
    times = np.asarray([time for _, time in rows], dtype=float)

    small = sizes <= SMALL_LIMIT
    large = sizes >= LARGE_LIMIT
    if not (np.any(small) and np.any(large)):
        raise ValueError(f"{path}: need small (<=64 B) and large (>=64 KiB) messages")

    t_s = float(np.median(times[small]))
    t_w = float(np.dot(sizes[large], times[large] - t_s) /
                np.dot(sizes[large], sizes[large]))
    if not np.isfinite(t_w) or t_w <= 0:
        raise ValueError(f"{path}: nonpositive fitted t_w; inspect CSV measurements")
    return Experiment(path, label, group, sizes, times, t_s, t_w)


def print_fit(exp):
    print(f"file={exp.path}")
    print(f"label={exp.label}")
    print(f"sizes={len(exp.sizes)}/{EXPECTED_SIZES}")
    print(f"t_s_us={exp.t_s * 1e6:.6f}")
    print(f"t_w_s_per_byte={exp.t_w:.12e}")
    print(f"bandwidth_MiB_per_s={exp.bandwidth_mib_s:.6f}")


def save_individual_plot(exp):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.loglog(exp.sizes, exp.times, "o-", label="Measured one-way time")
    ax.loglog(exp.sizes, exp.t_s + exp.sizes * exp.t_w, "--",
              label="Fitted latency + bandwidth model")
    ax.set(xlabel="Message size (bytes)", ylabel="Mean one-way time (s)",
           title=f"MPI ping-pong: {exp.label}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    dest = exp.path.with_suffix(".png")  # original file name preserved
    fig.savefig(dest, dpi=200)
    plt.close(fig)
    print(f"plot={dest}")


def save_comparison_plots(experiments, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    # Plot local SHM measurements first, then inter-VM measurements.
    experiments.sort(key=lambda exp: (exp.group, exp.label))

    fig, ax = plt.subplots(figsize=(12, 7))
    for exp in experiments:
        ax.loglog(exp.sizes, exp.times, "o-", markersize=3.5,
                  linewidth=1.6, label=exp.label)
    ax.set(xlabel="Message size (bytes, log₂ scale)",
           ylabel="Mean one-way communication time (seconds, log scale)",
           title="MPI ping-pong: shared memory vs inter-VM Tailscale")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = output_dir / "combined_oneway_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"plot={path}")

    fig, ax = plt.subplots(figsize=(12, 7))
    for exp in experiments:
        effective_bandwidth = exp.sizes / exp.times / (1024.0 ** 2)
        ax.plot(exp.sizes, effective_bandwidth, "o-", markersize=3.5,
                linewidth=1.6, label=exp.label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")  # SHM and inter-VM throughputs may differ by orders of magnitude
    ax.set(xlabel="Message size (bytes, log₂ scale)",
           ylabel="Effective payload bandwidth (MiB/s, log scale)",
           title="MPI ping-pong: effective bandwidth comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    path = output_dir / "combined_bandwidth_comparison.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"plot={path}")

    path = output_dir / "combined_fit_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "file", "sizes", "t_s_us",
                         "t_w_s_per_byte", "fitted_bandwidth_MiB_per_s"])
        for exp in experiments:
            writer.writerow([exp.label, exp.path.name, len(exp.sizes),
                             f"{exp.t_s * 1e6:.9f}", f"{exp.t_w:.12e}",
                             f"{exp.bandwidth_mib_s:.9f}"])
    print(f"summary={path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file", nargs="?", type=Path,
                        help="Original mode: fit one CSV and save its .png")
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--compare-dir", type=Path, metavar="DIRECTORY",
                         help="Auto-load SAME-VM SHM and INTER-VM CSVs (excludes TCP loopback)")
    options.add_argument("--compare", nargs="+", type=Path, metavar="CSV",
                         help="Explicit CSV files to compare")
    parser.add_argument("--output-dir", type=Path,
                        help="Where to save comparison PNGs and fit summary (default: CSV directory)")
    parser.add_argument("--individual", action="store_true",
                        help="Also regenerate a fitted .png for every compared CSV")
    args = parser.parse_args()

    if args.compare_dir is None and args.compare is None:
        if args.csv_file is None:
            parser.error("give a CSV file, --compare-dir DIRECTORY, or --compare CSV ...")
        if args.output_dir is not None or args.individual:
            parser.error("--output-dir and --individual require a comparison mode")
        exp = load_experiment(args.csv_file)
        print_fit(exp)
        save_individual_plot(exp)
        return

    if args.csv_file is not None:
        parser.error("do not combine the positional CSV with comparison options")

    if args.compare_dir is not None:
        directory = args.compare_dir
        if not directory.is_dir():
            parser.error(f"comparison directory not found: {directory}")
        paths = [p for p in directory.glob("ping_*.csv")
                 if classify_filename(p) is not None]
        paths.sort(key=lambda p: p.name)
        ignored = sorted(p.name for p in directory.glob("ping_*tcp*.csv"))
        if ignored:
            print("Ignored TCP loopback experiments: " + ", ".join(ignored))
        if not paths:
            parser.error(f"no SHM or inter-VM ping-pong CSVs found in {directory}")
        output_dir = args.output_dir or directory
    else:
        paths = args.compare
        output_dir = args.output_dir or paths[0].parent

    experiments = [load_experiment(path) for path in paths]
    if len(experiments) < 2:
        parser.error("comparison requires at least two experiment CSV files")
    print(f"Comparing {len(experiments)} experiments:")
    for exp in experiments:
        print(f"  {exp.label}: {len(exp.sizes)}/{EXPECTED_SIZES} sizes ({exp.path})")
        if len(exp.sizes) < EXPECTED_SIZES:
            print(f"  WARNING: {exp.path.name} is incomplete; comparison uses available sizes")
        if args.individual:
            print_fit(exp)
            save_individual_plot(exp)
    save_comparison_plots(experiments, output_dir)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
