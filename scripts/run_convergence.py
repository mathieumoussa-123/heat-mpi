#!/usr/bin/env python3
"""Part 1, Task 7.2: C++ serial-solver convergence vs the verified math PDF.

Run: python3 scripts/run_convergence.py --exe build/heat
Outputs: results/part1/task_7_2_convergence.csv and task_7_2_convergence.png
"""
import argparse
import csv
import math
from pathlib import Path
import subprocess

# Verified NUMPY results from our mathematical solutions PDF, Task 7.2.
# These are benchmarks for independently confirming the C++ implementation,
# NOT reused as the C++ experiment's measured errors.
MATH_REFERENCE = {
    16: (1.66337050e-3, 8.31685252e-4),
    32: (4.14182427e-4, 2.07091213e-4),
    64: (1.03442488e-4, 5.17212439e-5),
    128: (2.58541870e-5, 1.29270935e-5),
    256: (6.46314472e-6, 3.23157236e-6),
}


def parse_summary(stdout: str) -> dict[str, str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"expected one summary line, received: {stdout!r}")
    parts = dict(part.split("=", 1) for part in lines[0].split())
    for required in ("decomp", "case", "N", "r", "steps", "err_inf", "err_l2"):
        if required not in parts:
            raise ValueError(f"missing {required} in: {lines[0]}")
    return parts


def run(executable: Path, n: int) -> dict[str, str]:
    cmd = [str(executable), "--decomp", "serial", "--case", "V",
           "--N", str(n), "--r", "0.2", "--T", "0.05"]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return parse_summary(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/heat", help="path to compiled heat executable")
    parser.add_argument("--out-dir", default="results/part1", help="directory for CSV and PNG")
    args = parser.parse_args()

    executable = Path(args.exe).resolve()
    if not executable.exists():
        parser.error(f"executable not found: {executable}; build the project first")
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    records = []
    prev_inf = prev_l2 = None
    for n, reference in MATH_REFERENCE.items():
        summary = run(executable, n)
        actual_inf = float(summary["err_inf"])
        actual_l2 = float(summary["err_l2"])
        steps = int(summary["steps"])
        assert steps == n * n // 4, f"N={n}: expected {n*n//4} steps, received {steps}"
        for name, actual, expected in (("inf", actual_inf, reference[0]), ("l2", actual_l2, reference[1])):
            relative_error = abs(actual - expected) / expected
            if relative_error > 5e-6:
                raise AssertionError(f"N={n} {name}: C++={actual}, math PDF={expected}; rel={relative_error}")
        p_inf = math.log2(prev_inf / actual_inf) if prev_inf is not None else None
        p_l2 = math.log2(prev_l2 / actual_l2) if prev_l2 is not None else None
        if p_inf is not None and not (1.95 < p_inf < 2.05 and 1.95 < p_l2 < 2.05):
            raise AssertionError(f"unexpected convergence order at N={n}: {p_inf}, {p_l2}")
        record = dict(N=n, h=1 / n, steps=steps, err_inf=actual_inf, err_l2=actual_l2,
                      p_inf=p_inf, p_l2=p_l2, t_comp_s=float(summary["t_comp"]))
        records.append(record)
        print(f"N={n:3d}, steps={steps:5d}, err_inf={actual_inf:.9g}, "
              f"err_l2={actual_l2:.9g}, p_inf={p_inf if p_inf is not None else '-'}")
        prev_inf, prev_l2 = actual_inf, actual_l2

    csv_path = outdir / "task_7_2_convergence.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = [row["h"] for row in records]
    error_inf = [row["err_inf"] for row in records]
    error_l2 = [row["err_l2"] for row in records]
    reference_o2 = [error_inf[0] * (hi / h[0]) ** 2 for hi in h]

    plt.figure(figsize=(9, 6))
    plt.loglog(h, error_inf, marker="o", label=r"C++ $\|e\|_\infty$")
    plt.loglog(h, error_l2, marker="s", label=r"C++ $\|e\|_{2,h}$")
    plt.loglog(h, reference_o2, "--", label=r"Reference $O(h^2)$")
    plt.xlabel("Grid spacing h = 1/N")
    plt.ylabel("Error at T = 0.05")
    plt.title("Task 7.2 — Serial C++ convergence (r = 0.2)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path = outdir / "task_7_2_convergence.png"
    plt.savefig(png_path, dpi=180)
    plt.close()
    print(f"Reference check: PASS. Results: {csv_path} | {png_path}")


if __name__ == "__main__":
    main()
