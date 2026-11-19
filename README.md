# Heat MPI cluster setup starter

See `GROUP_SETUP_GUIDE.md` for the complete group instructions.

This starter contains *only Part 0 setup utilities*, not the final heat solver:
- `src/mpi_hello.cpp` — verifies distributed MPI ranks.
- `src/mpi_pingpong.cpp` — times round trips from 8 B through 16 MiB and records batch variability (default 3 batches).
- `scripts/fit_pingpong.py` — reports the one-way latency / effective bandwidth model.
- `AGENTS.md` and `llm_log/` — helps fulfill course logging requirements.
- `hosts.example` — replace names and process slots.

Run on *every* Ubuntu VM, in the same checkout path:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

For the actual benchmark commands, read `GROUP_SETUP_GUIDE.md`.

The starter code is illustrative; validate it with `mpirun` on your own cluster,
record failures, and explain its operation. The professor requires all members to
understand and take responsibility for the code they submit.

Benchmark update: use `--mca pml ob1 --mca btl self,vader` for same-VM shared
memory on Ubuntu 24.04 / Open MPI 4.x, and `--mca pml ob1 --mca btl self,tcp`
plus the Tailscale interface selectors for cross-VM tests. See guide Section 3.
The first four CSV columns are backward compatible with the fitting script.

## Part 1 — Serial heat solver (implemented)

See [`PART1_SERIAL_GUIDE.md`](PART1_SERIAL_GUIDE.md) for full commands, design and requirements.

Build on each Ubuntu VM (same procedure as Part 0):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Run an ordinary stable case or verification case:

```bash
./build/heat --decomp serial --case 1 --N 128 --r 0.2 --steps 500 --out results/part1/plate_case1.bin
./build/heat --decomp serial --case V --N 64 --r 0.2 --T 0.05
python3 scripts/test_serial.py --exe build/heat
python3 scripts/run_convergence.py --exe build/heat
python3 scripts/run_instability.py --exe build/heat
```

The serial `heat` executable does not use MPI. Part 0's `mpi_hello` and `mpi_pingpong`
programs and comparison scripts continue to work unchanged.
