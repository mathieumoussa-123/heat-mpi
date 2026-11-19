# Part 1 — Serial solver implementation and evidence

Based directly on `heat_mpi_project.pdf` §§3,5 and our mathematical solutions
(Tasks 5.2, 5.5, 6.3, 6.7, 7.2, 8.2). Part 0 remains intact.

## Included files

- `src/heat.hpp` — shared config, array index, entry points for future MPI versions.
- `src/heat_main.cpp` — full CLI, stability checks, summary line.
- `src/heat_serial.cpp` — initialization, two-array time stepping, error norms.
- `src/heat_io.cpp` — raw IEEE-754 little-endian binary grid output.
- `scripts/test_serial.py` — short boundary/CLI/format regression tests.
- `scripts/run_convergence.py` — Task 7.2 C++ measurement, table, convergence PNG.
- `scripts/run_instability.py` — Task 6.7 5%-above-limit growth and snapshots.
- `CMakeLists.txt` — adds heat while preserving MPI Part 0 targets.

## Build on Mathieu, Naji, and Chris VMs

```bash
cd /home/mpi/heat-mpi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/test_serial.py --exe build/heat
```

Requires `build-essential cmake openmpi-bin libopenmpi-dev python3-numpy python3-matplotlib`
(the same packages as the project PDF's Part 0). The serial `heat` process must NOT
be launched via `mpirun`; `P=1` and `t_comm=0` by definition. Compile it on every
VM to establish machine-specific performance baselines for Part 4.

## CLI examples

```bash
mkdir -p results/part1
./build/heat --case 1 --N 128 --r 0.2 --steps 500 --decomp serial --out results/part1/case1.bin
./build/heat --case 2 --N 128 --r 0.2 --steps 500 --decomp serial --out results/part1/case2.bin
./build/heat --case V --N 64 --r 0.2 --T 0.05 --decomp serial
```

- `--N`: N intervals per side, N+1 nodes, grid spacing h=1/N.
- `--r`: requested r=alpha*dt/h^2, alpha=1. Reject r<=0 or r>0.25.
- `--steps`: fixed number of time steps; default 100 if neither --steps nor --T provided.
- `--T`: choose a positive integer step count s.t. dt=T/steps, and effective r<=requested r.
  `--steps` and `--T` cannot be supplied together.
- `--case`: 1 (all edges 1), 2 (top and bottom 1, left/right 0 except corners),
  V (homogeneous edges, sine-sine exact solution).
- `--decomp serial`: retained as the reference for upcoming strips/blocks.
- `--out`: optional raw little-endian double grid, **no header**, shape (N+1)x(N+1),
  j rows (y increases), i varies fastest within a row (x increases).

Algorithm is Task 5.2(b):

```
u_new[i,j] = u_old[i,j] + r * (u_old[i-1,j] + u_old[i+1,j]
                                 + u_old[i,j-1] + u_old[i,j+1] - 4*u_old[i,j])
```

Two flat std::vector<double> arrays are swapped (never copied); the timing of the
`std::chrono::steady_clock` update loop excludes initialization, error calculations,
and file output. Compilation uses C++17, `-O2 -ffp-contract=off`.

Standard output is a single machine-parseable line:

```
decomp=serial P=1 case=V N=64 r=... requested_r=... dt=... T=... steps=... t_total=... t_comp=... t_comm=0 err_inf=... err_l2=...
```

The verification field is `sin(pi*x)*sin(pi*y)*exp(-2*pi*pi*T)`;
`err_inf` is the maximum pointwise absolute difference, and
`err_l2` is `sqrt(h*h*sum_{i,j} error[i,j]^2)` including boundaries.
Case 1/2 do not print error norms because no exact solution is supplied.

## Required Task 7.2 convergence (C++ runs, not reusing NumPy measurements)

```bash
python3 scripts/run_convergence.py --exe build/heat
```

This uses the **exact parameters of our mathematical PDF**: alpha=1, T=0.05,
requested r=0.2, and N=16,32,64,128,256 (steps N^2/4).
The script runs each C++ simulation and verifies its errors against the APPROVED
NumPy mathematical solutions within a small floating-point tolerance; the CSV
records C++ results only. Expected observed order is about 2.

Outputs:
- `results/part1/task_7_2_convergence.csv`
- `results/part1/task_7_2_convergence.png`

## Required Task 6.7 instability (deliberately violate stability limit)

```bash
python3 scripts/run_instability.py --exe build/heat
```

Follows our mathematical Task 6.7 precisely: N=64, 240 steps, r=0.25 and
r=0.2625 (5% above the limit), initial field
`sin(pi*x)*sin(pi*y) + 1e-6*sin((N-1)*pi*x)*sin((N-1)*pi*y)`.
Snapshots and max|u| are from the C++ binary (initial n=0 from the prescribed formula).

Ordinary runs ALWAYS reject `r>0.25` with an error. To reproduce the required
unstable experiment explicitly, use the flagged command below:

```bash
./build/heat --decomp serial --case V --N 64 --r 0.2625 --steps 240 \
  --seed-high-mode 1e-6 --allow-unstable --out results/part1/unstable_final.bin
```

`--allow-unstable` is intentionally constrained to Case V, seeded mode, and
`--steps`; it also prints a warning to stderr. Never use it for Cases 1/2
or production performance experiments.

Outputs:
- `results/part1/task_6_7_instability.csv`
- `results/part1/task_6_7_growth.png`
- `results/part1/task_6_7_snapshots.png`

## Validation before committing

```bash
ctest --test-dir build --output-on-failure
python3 scripts/test_serial.py --exe build/heat
python3 scripts/run_convergence.py --exe build/heat
python3 scripts/run_instability.py --exe build/heat
```

Check generated PNGs. Do not commit large .bin files unless specifically requested.
The source and plotted outputs were generated locally for verification; rerun them
on the actual Ubuntu VMs and save your team-specific timings/results.

## Git and course AI logging

This ZIP is a **repository snapshot**, not a live modification of your GitHub repo.
Integrate `src/heat*`, the new `scripts/` files, `CMakeLists.txt`, README additions,
and `PART1_SERIAL_GUIDE.md` into your EXISTING checkout without deleting your
`results/`, `llm_log/`, Git history, or teammates' modifications.

**Professor's AI rules:** export your full chat transcript and model/tool info,
append the required records to `llm_log/`, note all edited files, and commit the
full logs alongside source. This artifact cannot automatically capture every
prior chat interaction, so no claim of complete logging is made.
