#!/usr/bin/env python3
"""Part 1, Task 6.7: reproduce instability 5% beyond the CFL limit.

Run: python3 scripts/run_instability.py --exe build/heat
Generates measured max|u| CSV, growth plot, and temperature snapshot PNG.
"""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np

STEPS = (0, 40, 80, 120, 160, 200, 240)
# Rounded reference measurements from our math solutions PDF, Task 6.7.
MATH_REFERENCE = {
    0.25: {0: 1.000001, 40: 0.9529340, 80: 0.9080820, 120: 0.8653410,
           160: 0.8246120, 200: 0.7858000, 240: 0.7488147},
    0.2625: {0: 1.000001, 40: 0.9506800, 80: 0.9055790, 120: 0.9398550,
             160: 4.307196, 200: 151.6488, 240: 6522.004},
}
N = 64
AMPLITUDE = 1e-6
R_VALUES = (0.25, 0.2625)


def initial_grid() -> np.ndarray:
    coords = np.arange(N + 1, dtype=np.float64) / N
    x, y = np.meshgrid(coords, coords)
    u = np.sin(np.pi * x) * np.sin(np.pi * y)
    u += AMPLITUDE * np.sin((N - 1) * np.pi * x) * np.sin((N - 1) * np.pi * y)
    u[0, :] = u[-1, :] = 0
    u[:, 0] = u[:, -1] = 0
    return u


def simulate(exe: Path, r: float, steps: int, output: Path) -> np.ndarray:
    cmd = [str(exe), "--decomp", "serial", "--case", "V", "--N", str(N),
           "--r", str(r), "--steps", str(steps), "--seed-high-mode", str(AMPLITUDE),
           "--out", str(output)]
    if r > 0.25:
        cmd.append("--allow-unstable")
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = np.fromfile(output, dtype="<f8")
    if data.size != (N + 1) ** 2:
        raise RuntimeError(f"incorrect output size for {r=}, {steps=}: {data.size}")
    return data.reshape((N + 1, N + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/heat")
    parser.add_argument("--out-dir", default="results/part1")
    args = parser.parse_args()
    exe = Path(args.exe).resolve()
    if not exe.exists():
        parser.error(f"executable not found: {exe}")
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    grids = {}
    rows = []
    with tempfile.TemporaryDirectory(prefix="heat_instability_") as workdir:
        binary_path = Path(workdir) / "grid.bin"
        for r in R_VALUES:
            for step in STEPS:
                # n=0: initial condition prescribed analytically, no solver steps taken.
                field = initial_grid() if step == 0 else simulate(exe, r, step, binary_path)
                maximum = float(np.max(np.abs(field)))
                expected = MATH_REFERENCE[r][step]
                if abs(maximum - expected) / max(1.0, abs(expected)) > 5e-5:
                    raise AssertionError(
                        f"Task 6.7 mismatch for r={r}, step={step}: "
                        f"C++={maximum}, math PDF={expected}"
                    )
                grids[(r, step)] = field
                rows.append(dict(r=r, step=step, max_abs_u=maximum,
                                 origin="prescribed initial condition" if step == 0 else "C++ serial solver"))
                print(f"r={r:.4f}, step={step:3d}, max_abs_u={maximum:.9g}")

    csv_path = outdir / "task_6_7_instability.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    for r in R_VALUES:
        subset = [record for record in rows if record["r"] == r]
        plt.semilogy([record["step"] for record in subset],
                     [record["max_abs_u"] for record in subset],
                     marker="o", label=f"r = {r}")
    plt.xlabel("Time-step index")
    plt.ylabel(r"Maximum absolute grid temperature $\max|u|$")
    plt.title("Task 6.7 — Stable vs unstable explicit scheme")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    growth_path = outdir / "task_6_7_growth.png"
    plt.savefig(growth_path, dpi=180)
    plt.close()

    show_steps = (0, 80, 160, 240)
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    for k, r in enumerate(R_VALUES):
        for col, step in enumerate(show_steps):
            field = grids[(r, step)]
            ax = axes[k, col]
            im = ax.imshow(field, origin="lower", extent=(0, 1, 0, 1), interpolation="nearest")
            ax.set_title(f"r={r}, step={step}\nmax|u|={np.max(np.abs(field)):.3g}")
            ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Task 6.7 — Temperature snapshots (C++ solver, except analytic initial field)")
    snaps_path = outdir / "task_6_7_snapshots.png"
    fig.savefig(snaps_path, dpi=160)
    plt.close(fig)
    if grids[(0.2625, 240)].max() < 100:
        raise AssertionError("expected strong instability at r=0.2625, step=240")
    print(f"Math PDF reference check: PASS. Saved: {csv_path} | {growth_path} | {snaps_path}")


if __name__ == "__main__":
    main()
