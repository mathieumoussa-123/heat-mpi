#!/usr/bin/env python3
"""Fast project-specific regression checks for Part 1 (no pytest needed)."""
import argparse
import math
import struct
import subprocess
import tempfile
from pathlib import Path


def call(executable: Path, *args: str, ok: bool = True) -> dict[str, str]:
    process = subprocess.run([str(executable), *args], capture_output=True, text=True)
    if ok and process.returncode != 0:
        raise AssertionError(f"failed: {args}\n{process.stderr}")
    if not ok:
        assert process.returncode != 0, f"expected rejection: {args}"
        return {"stderr": process.stderr}
    parts = process.stdout.strip().split()
    return dict(part.split("=", 1) for part in parts)


def field(path: Path, n: int) -> tuple[float, ...]:
    data = path.read_bytes()
    assert len(data) == 8 * (n + 1) ** 2, "incorrect binary format or dimensions"
    return struct.unpack("<" + "d" * ((n + 1) ** 2), data)


def idx(i: int, j: int, n: int) -> int:
    return j * (n + 1) + i


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", default="build/heat")
    args = parser.parse_args()
    exe = Path(args.exe).resolve()
    assert exe.is_file(), f"compile heat first: {exe}"
    with tempfile.TemporaryDirectory(prefix="part1_test_") as temp:
        folder = Path(temp)
        n = 8
        p1 = folder / "case1.bin"
        c1 = call(exe, "--case", "1", "--N", str(n), "--r", "0.2", "--steps", "1", "--out", str(p1))
        u1 = field(p1, n)
        assert c1["decomp"] == "serial" and c1["P"] == "1"
        assert all(u1[idx(i, 0, n)] == 1 for i in range(n + 1))
        assert all(u1[idx(0, j, n)] == 1 for j in range(n + 1))
        assert abs(u1[idx(1, 1, n)] - 0.4) < 1e-15, "case 1: first interior update should be 2r"
        assert u1[idx(4, 4, n)] == 0.0, "far-away interior remains cold after 1 step"

        p2 = folder / "case2.bin"
        call(exe, "--case", "2", "--N", str(n), "--r", "0.2", "--steps", "1", "--out", str(p2))
        u2 = field(p2, n)
        assert all(u2[idx(i, 0, n)] == 1.0 and u2[idx(i, n, n)] == 1.0 for i in range(n + 1))
        assert all(u2[idx(0, j, n)] == 0.0 and u2[idx(n, j, n)] == 0.0 for j in range(1, n))
        assert abs(u2[idx(1, 1, n)] - 0.2) < 1e-15
        assert u2[idx(4, 4, n)] == 0.0

        # Task 7.2 N=16, exact same output for steps vs --T.
        p_steps, p_time = folder / "steps.bin", folder / "time.bin"
        a = call(exe, "--case", "V", "--N", "16", "--r", "0.2", "--steps", "64", "--out", str(p_steps))
        b = call(exe, "--case", "V", "--N", "16", "--r", "0.2", "--T", "0.05", "--out", str(p_time))
        assert int(b["steps"]) == 64 and abs(float(b["T"]) - 0.05) < 1e-14
        assert p_steps.read_bytes() == p_time.read_bytes(), "--T changed the numerical update"
        assert abs(float(a["err_inf"]) - 1.66337050e-3) < 1e-10

        # Choose a noninteger T/dt: requested r must not be exceeded.
        t = call(exe, "--case", "V", "--N", "16", "--r", "0.2", "--T", "0.053")
        assert abs(float(t["T"]) - 0.053) < 1e-14
        assert float(t["r"]) <= 0.2 * (1 + 1e-14)
        assert int(t["steps"]) >= math.ceil(0.053 * 16 * 16 / 0.2 - 1e-10)

        # No unstable ordinary runs. Intentional Task 6.7 is a documented exception.
        refused = call(exe, "--case", "V", "--N", "64", "--r", "0.2625",
                       "--steps", "240", ok=False)
        assert "unstable" in refused["stderr"]
        call(exe, "--case", "V", "--N", "64", "--r", "0.2625", "--steps", "40",
             "--seed-high-mode", "1e-6", "--allow-unstable")
        call(exe, "--case", "V", "--N", "64", "--r", "0.25", "--steps", "40",
             "--seed-high-mode", "1e-6")
        call(exe, "--case", "2", "--N", "8", "--T", "0.05", "--r", "0.25")
        call(exe, "--decomp", "unrecognized", ok=False)
        call(exe, "--r", "0", ok=False)
        call(exe, "--T", "-1", ok=False)
        call(exe, "--steps", "64", "--T", "0.05", ok=False)
    print("PASS: cases 1/2/V, boundaries, stencil, output, timing metadata, --T, stability gate")


if __name__ == "__main__":
    main()
