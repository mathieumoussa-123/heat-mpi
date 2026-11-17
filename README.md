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
