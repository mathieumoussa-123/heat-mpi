# Part 2 — MPI 1D strip decomposition: implementation and verification

**Basis:** `heat_mpi_project.pdf`, Part 2, §§6.1–6.3 and Checkpoint C2; the project's approved mathematical-solution PDF, especially Task 5.2 (five-point scheme), Task 6.3 (stability), Task 8.1 (why two arrays), and Task 8.2 (halo costs). This adds to the working Part 1 code, not a replacement for it.

## What is implemented

- The **same** executable `build/heat` has `--decomp serial` and `--decomp strips`; blocks are deferred to Part 3.
- `N` is **intervals**, hence the full field is `(N+1) x (N+1)` including the physical boundaries.
- Interior rows `1..N-1` are partitioned in continuous strips across MPI ranks `0..P-1`, bottom to top. If `q=(N-1)/P` and `s=(N-1)%P`, then rank `p` owns `q + (p<s)` rows and its first global row is `1 + p*q + min(p,s)`.
- Every rank allocates **two arrays** of `(n_local+2)*(N+1)` doubles. Local row `0` is the bottom ghost, rows `1..n_local` are owned, row `n_local+1` is the top ghost. Rank 0's bottom ghost and rank P-1's top ghost are fixed physical-boundary rows.
- At the **start of each step**, two `MPI_Sendrecv` operations transfer the top/bottom owned rows (each `N+1` `MPI_DOUBLE`, no packing). Missing outside neighbors are `MPI_PROC_NULL`.
- Only nodes with local `j=1..n_local` and global `i=1..N-1` are updated, using **the exact same floating-point stencil arithmetic** as Part 1, and then the two vectors are swapped.
- One `MPI_Barrier` occurs before timing. `MPI_Wtime` measures only the stepping loop. Max `t_total`, max `t_comp` and max `t_comm` are reduced independently with `MPI_Reduce` / `MPI_MAX` and printed only on rank 0.
- If `--out` is given (or for verification Case V), `MPI_Gatherv` reconstructs owned interior rows in row-major order at rank 0, which fills the two global boundary rows, reuses Part 1's raw little-endian binary writer, and calculates the same Case V error norms. Gathering and writing are **not** timed. Without `--out` and outside Case V, the field is **not gathered**, avoiding unnecessary memory/network costs for timing runs.
- `--T` uses the existing Part 1 rule: integer number of steps, actual `r` no larger than requested. `r>0.25` is forbidden in ordinary runs. Task 6.7's intentionally unstable test-only flag remains supported.
- Reject `P>N-1` explicitly (empty-strip ranks not supported). Regular C2 checks only use `P<=N-1`.
- `--debug-halos` (strips only, `N<=16`) performs **untimed** initial and final exchanges and prints all local values including ghost rows to stderr, labeled `stage=initial` and `stage=final` and sorted by rank. Useful for the professor's debugging example `N=8`, one step, P=2 or P=3. Halo printing is intended for correctness checks, not speed measurements.

### Why `MPI_Sendrecv` matters

If every rank calls `MPI_Send` to its upper neighbor **before** posting its matching `MPI_Recv`, eager buffering may allow small transfers to proceed, but once messages exceed the eager limit, blocking sends may wait on a receive that has not yet been posted. This can deadlock. `MPI_Sendrecv` posts complementary send and receive as a single matching operation, so this cyclic waiting pattern does not occur. Rank order and distinct tags (`71` for upwards, `72` for downwards) keep the two exchanges matched.

### Mathematical context

The explicit scheme is

`u_new(i,j) = u_old(i,j) + r * ((u_old(i-1,j) + u_old(i+1,j) + u_old(i,j-1) + u_old(i,j+1)) - 4*u_old(i,j))`,

where `r=alpha*dt/h^2`, `alpha=1`, `h=1/N`, `r<=1/4`. Task 8.1 proves why the update cannot overwrite the old time level. Task 8.2 predicts that an interior strip communicates two halo rows per step of length roughly `N` (the actual project grid sends `N+1` doubles including boundary columns), giving an approximate cost of `2*t_s + 2*(N+1)*t_w` under the one-way-message modeling convention. Real latency over Tailscale is measured in Part 0; it is **not** assumed in this code.

## Install on all three Ubuntu VMs

**Before applying overlay**: commit or back up local modifications to `CMakeLists.txt`, `src/heat_main.cpp`, `src/heat.hpp`, `README.md` or `scripts/test_serial.py`. The overlay replaces these files. Do not blindly overwrite teammates' edits.

1. Download `Heat_MPI_Part2_Overlay.zip` on your laptop.
2. Copy to Mathieu's VM (Windows PowerShell example):

```powershell
scp "$HOME\Downloads\Heat_MPI_Part2_Overlay.zip" mpi@heat-mathieu:/home/mpi/
```

3. On Mathieu:

```bash
cd /home/mpi/heat-mpi
git status
unzip -o /home/mpi/Heat_MPI_Part2_Overlay.zip -d .
git diff
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/test_serial.py --exe build/heat
```

4. Run the *local MPI correctness matrix* on Mathieu:

```bash
mkdir -p results/part2
set -o pipefail
python3 scripts/test_strips.py --exe build/heat \
    --report results/part2/strips_local_mathieu.csv \
    | tee results/part2/strips_local_mathieu.log
```

The full matrix checks P=1,2,3,4 across 7 combinations including cases 1/2/V, uneven row counts, the mandatory one-step debugging grid, a noninteger `--T`, and multiple grid sizes. The expected final message is `PASS: 28 byte-for-byte strip comparisons`.

5. After reviewing your Git diff, update the **complete** required `llm_log/` conversation record; then commit and push:

```bash
git add CMakeLists.txt README.md PART2_STRIPS_GUIDE.md
git add src/heat.hpp src/heat_main.cpp src/heat_strips.cpp
git add scripts/test_serial.py scripts/test_strips.py llm_log/
git commit -m "Implement Part 2 MPI strip solver and correctness tests"
git push origin main
```

6. On **Naji and Chris separately**, pull the pushed source and compile locally:

```bash
cd /home/mpi/heat-mpi
git status
git pull --ff-only origin main
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/test_serial.py --exe build/heat
```

Optional: also run the local strip test script independently on Naji/Chris, and save logs with their VM names. No one needs to rerun Part 0 ping-pong measurements.

## First remote MPI comparison — Mathieu ↔ Naji

**Make sure both VMs have the binary at `/home/mpi/heat-mpi/build/heat` and passwordless SSH works.** From Mathieu:

```bash
cd /home/mpi/heat-mpi
mkdir -p results/part2
./build/heat --decomp serial --case 2 --N 8 --r 0.2 \
  --steps 1 --out results/part2/serial_case2_N8.bin

mpirun --host heat-mathieu:1,heat-naji:1 --map-by node -np 2 \
  --mca pml ob1 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl self,tcp \
  /home/mpi/heat-mpi/build/heat \
  --decomp strips --case 2 --N 8 --r 0.2 --steps 1 \
  --debug-halos --out /home/mpi/heat-mpi/results/part2/strips_case2_N8.bin \
  2>&1 | tee results/part2/debug_P2_N8.log

cmp results/part2/serial_case2_N8.bin results/part2/strips_case2_N8.bin
```

`cmp` emits **no output** on equality (exit status zero). The top and bottom physical boundaries should appear correctly in ghost rows in the debug log.

Try a three-rank run on all VMs (N=8, uneven: 3/2/2 rows):

```bash
mpirun --host heat-mathieu:1,heat-naji:1,heat-chris:1 --map-by node -np 3 \
  --mca pml ob1 --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 --mca btl self,tcp \
  /home/mpi/heat-mpi/build/heat \
  --decomp strips --case 2 --N 8 --r 0.2 --steps 1 \
  --debug-halos --out /home/mpi/heat-mpi/results/part2/strips_case2_N8_P3.bin \
  2>&1 | tee results/part2/debug_P3_N8.log
cmp results/part2/serial_case2_N8.bin results/part2/strips_case2_N8_P3.bin
```

**Important:** the full output file is written by MPI rank 0, which in these examples is intended to be on Mathieu. If you change host ordering and rank 0 goes to another VM, the output path must exist **there**, and the `cmp` must run where the output resides.

## Automated remote Checkpoint C2

The test script supports the Open MPI host list or a hostfile, and checks the full serial vs MPI **file bytes** (`sha256`, P, output size, Case V error fields). For initial remote runs, limit to the quick matrix to avoid long waits due to observed Tailscale RTTs:

```bash
cd /home/mpi/heat-mpi
set -o pipefail
python3 scripts/test_strips.py --exe build/heat \
  --hosts heat-mathieu:2,heat-naji:2,heat-chris:2 \
  --tailscale --quick --ranks 1,2,3,4 \
  --report results/part2/strips_C2_distributed.csv \
  | tee results/part2/strips_C2_distributed.log
```

This validates **12 combinations**. The `--map-by node` flag built into the script spreads ranks across machines when P>=2. A full C2 matrix drops `--quick` and runs **28 combinations**. For all inter-VM tests ensure Mathieu is the first listed host, so that rank 0 can write into the temporary output directory created on Mathieu; the other VMs don't need that temporary directory. Host slots must reflect actual vCPUs. You can also use `--hostfile hosts` instead of `--hosts ...` (not both).

The three VM correctness runs should be repeated for P=1,2,3,4 and several N (including N-1 not divisible by P) until all pass. Save the CSV and logs in `results/part2/` and preserve the exact CLI environment and hostname list. For mixed x86-64 and ARM clusters, run same-architecture bitwise comparisons and separately report the largest absolute difference on mixed-architecture runs, as the PDF requires. Use `--mixed-arch-report` to continue through the whole matrix and record `max_abs_diff` in the CSV even if cross-architecture bytes differ; this does **not** count as passing the normal bitwise check. On x86-64-only VMs, every C2 binary comparison must pass.

## Output, timing and traps

The `heat` summary is one line printed **only by rank 0**, e.g.:

`decomp=strips P=3 case=V N=16 r=0.2 requested_r=0.2 dt=... T=... steps=64 t_total=... t_comp=... t_comm=... err_inf=... err_l2=...`

- `t_total`: max elapsed step-loop duration over all ranks.
- `t_comp`: max sum of local stencil/update durations over all ranks.
- `t_comm`: max sum of local `MPI_Sendrecv` durations over all ranks. **The three maxima are reduced independently**, so `t_comp + t_comm` need not equal `t_total`.
- `--out`: on rank 0, writes exactly `(N+1)^2 * 8` raw bytes in row-major order, little-endian, without header; the full perimeter is included.
- Debug printing is untimed but requires extra initial/final halo exchanges, so avoid `--debug-halos` during performance measurements.
- `P>N-1` is intentionally rejected (every rank must own at least one interior row). Use `N>=P+1`.
- For `r>0.25` ordinary runs must reject. The existing `--allow-unstable` remains a deliberately labeled test-only exception for Task 6.7.
- For full performance measurements and MPI scaling, follow the professor's **Part 4**: repeat at least three times, report median and spread, control vCPU over-subscription, and distinguish `--map-by node` from `--map-by slot`.

## Verification boundary

The implementation was checked locally against the Part 1 numerical reference using an **MPI protocol emulator** because Open MPI runtime/header is not installed in the code-generation environment. That is **not equivalent** to testing with real Open MPI. The professor's Part 2 Checkpoint C2 requires successful execution and binary comparisons on the actual Ubuntu VMs, with at least two different VMs participating. Only call Part 2 fully verified after those real tests pass.

## LLM log requirement

Commit a **complete transcript** of this interaction (including full prompts, responses, timestamps, model, team member, and files modified). The bundled `llm_log/PART2_REVIEW_CHECKLIST.md` is only a checklist, NOT a transcript and NOT a substitute for the required raw exported history. Do not put passwords or SSH keys in logs.
