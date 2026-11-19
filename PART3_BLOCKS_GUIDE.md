# Part 3 — MPI Cartesian 2D blocks (implementation and validation)

**Project source:** `heat_mpi_project.pdf`, Section 7 (Part 3), alongside the program interface/output rules in Section 3 and Part 4 measurement rules. **Mathematical reference:** `Heat_Equation_Mathematical_Tasks.pdf`, especially Tasks 5.2, 6.3, 8.1 and 8.2.

The new `--decomp blocks` implementation extends the **same executable** used in Parts 1 and 2. The serial reference (`--decomp serial`) and the strip implementation (`--decomp strips`) keep their existing arithmetic and file layout.

## 1. Professor's requirements and how they are met

| Requirement | Implementation |
|---|---|
| `MPI_Dims_create` | Automatically chooses a 2D process grid: `dims[0]=Py`, `dims[1]=Px`. |
| Nonperiodic `MPI_Cart_create` | `periods={0,0}`, `reorder=0` so world rank 0 remains the output root. |
| `MPI_Cart_shift` | Finds north/south and east/west neighbors; physical edges receive `MPI_PROC_NULL`. |
| Balanced rectangular blocks | Split the `N-1` interior indices separately in x and y using quotient/remainder, with extras on lower process coordinates. |
| Four-sided ghosts | Each rank allocates `(nx+2)*(ny+2)` doubles for each time level. |
| Contiguous rows | Send `nx` `MPI_DOUBLE` values directly. |
| Noncontiguous columns | Send one `MPI_Type_vector(ny,1,nx+2,MPI_DOUBLE)` per column, committed and freed; **no column packing**. |
| Preserve five-point stencil | Same operation order: left+right+bottom+top, then `u + r*(sum-4*u)`; two arrays swapped. |
| Output identical to serial | Gather packed final interior tiles using `MPI_Gatherv`; rank 0 restores boundary nodes and writes raw little-endian `(N+1)x(N+1)` doubles. |
| Correct timing | `MPI_Wtime` for time-stepping only, separate compute/communication, reduce each using `MPI_MAX`. |
| Task 8.2 prediction & checking | `scripts/compare_decomps.py` uses corrected Part 0 fitted `t_s` and `t_w`; optionally launches measured P=4,8 / N=512,2048,8192 experiments. |

Corners of a block are *not* exchanged. The five-point stencil only reads the four cardinal directions. A nine-point stencil using diagonal values would need corner ghost exchanges.

### Memory, indexing and boundaries

Interior ownership is always `1 <= global_i,global_j <= N-1`. For a process coordinate `c` along an axis with `Q` process coordinates, it owns:

```
base   = (N-1) / Q
extra  = (N-1) % Q
count  = base + (c < extra ? 1 : 0)
first  = 1 + c*base + min(c,extra)
```

Local indices `i=1..nx`, `j=1..ny` are owned; `i=0,nx+1`, `j=0,ny+1` are ghosts. `MPI_Type_vector(ny,1,nx+2,...)` sends/receives one noncontiguous interior column. Ghost corners never participate in any stencil operation.

The upper and lower physical edges are hot for Case 2, **including the four corners**, matching the Part 1 reference exactly. Outer left/right edges are cold for Case 2 except those hot corners; all edges are hot for Case 1 and zero for Case V.

## 2. Install on Mathieu (the Git owner / launcher)

1. Download **`Heat_MPI_Part3_Overlay.zip`** to your computer.
2. Copy it to Mathieu from Windows PowerShell:

```powershell
scp "$HOME\Downloads\Heat_MPI_Part3_Overlay.zip" mpi@heat-mathieu:/home/mpi/
```

3. On Mathieu:

```bash
cd /home/mpi/heat-mpi
git status
unzip -o /home/mpi/Heat_MPI_Part3_Overlay.zip -d .
git diff
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/test_serial.py --exe build/heat
```

Review the diff, particularly `heat_main.cpp`, `CMakeLists.txt`, and `README.md`, in case teammates independently edited them. If a 3/4-rank *manual local MPI* command complains about slots, use `--host localhost:4` (four configured vCPUs per VM). `ctest` only runs P=1/2 smoke tests.

## 3. Validate MPI blocks locally — on all three VMs

On each VM after building:

```bash
cd /home/mpi/heat-mpi
mkdir -p results/part3
set -o pipefail
python3 scripts/test_blocks.py \
  --exe build/heat \
  --hosts localhost:4 \
  --ranks 1,2,3,4 \
  --report results/part3/blocks_local_$(hostname).csv \
  | tee results/part3/blocks_local_$(hostname).log
```

Expected **32** successful byte-for-byte serial/blocks comparisons (8 scenarios × four rank counts). Use a **different CSV filename** for reruns: the script intentionally refuses to append duplicate results to an existing report.

Optional stronger three-way comparison (more MPI runs):

```bash
python3 scripts/test_blocks.py \
  --exe build/heat --hosts localhost:4 --quick --also-strips \
  --report results/part3/blocks_threeway_local_$(hostname).csv
```

For visual debugging on tiny grids:

```bash
mpirun --host localhost:4 -np 4 /home/mpi/heat-mpi/build/heat \
  --decomp blocks --case 2 --N 8 --r 0.2 --steps 2 \
  --debug-halos
```

Debug output shows initial and final local matrices, including ghost layers, before/after timing. The four corner ghosts are not transferred.

## 4. Distributed two-VM acceptance — actual column and row exchange

First build **the same source on both VMs at the same absolute path**, then run on Mathieu:

```bash
cd /home/mpi/heat-mpi
mkdir -p results/part3
./build/heat --decomp serial --case 2 --N 8 --r 0.2 --steps 9 \
  --out results/part3/serial_2vm.bin

mpirun --host heat-mathieu:2,heat-chris:2 --map-by node -np 4 \
  --mca pml ob1 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl self,tcp \
  /home/mpi/heat-mpi/build/heat \
  --decomp blocks --case 2 --N 8 --r 0.2 --steps 9 \
  --out /home/mpi/heat-mpi/results/part3/blocks_2vm.bin

cmp results/part3/serial_2vm.bin results/part3/blocks_2vm.bin
```

`cmp` **must produce no output and exit 0**. This P=4 test creates a true 2×2 process grid and exercises column as well as row ghosts across two hosts. Only MPI rank zero writes the output file, so the result is on Mathieu.

## 5. Full three-VM correctness evidence

After Mathieu has pushed the Part 3 source and Naji/Chris have pulled and rebuilt, run on Mathieu:

```bash
cd /home/mpi/heat-mpi
set -o pipefail
python3 scripts/test_blocks.py \
  --exe build/heat \
  --hosts heat-mathieu:4,heat-naji:4,heat-chris:4 \
  --tailscale --quick --ranks 1,2,3,4,6,8 \
  --report results/part3/blocks_distributed_quick.csv \
  | tee results/part3/blocks_distributed_quick.log
```

Expected **18** block comparisons (3 cases × 6 process counts). `P=6` tests 3×2; `P=8` tests 4×2. These are real MPI runs. The given `:4` slots permit P up to 12 across the cluster (never >4 ranks per VM). For the **full** matrix, omit `--quick`, use a new filename, and expect **48** comparisons with this rank list.

For a smaller subset, run `--ranks 1,2,3,4` and expect 12 quick / 32 full comparisons. `--also-strips` additionally verifies three-way equality (expensive, optional); tests where P>N-1 skip the strip comparison.

Confirm placement separately with:

```bash
mpirun --host heat-mathieu:4,heat-naji:4,heat-chris:4 --map-by node -np 8 hostname
```

The `--hosts` list alone does not prove exactly which rank ran on which host; retain this placement output for the report. For TCP over Tailscale, ensure `tailscale0` exists and the SSH keys/permissions still work.

## 6. Prediction and performance comparison (Part 3 requirement)

The professor requests **P=4,8** and **N=512,2048,8192**, predicting then checking which decomposition is faster. First run the **safe, prediction-only** command on Mathieu:

```bash
cd /home/mpi/heat-mpi
python3 scripts/compare_decomps.py \
  --fit-summary data/part0_corrected_fit_summary.csv \
  --out-dir results/part3
```

This generates `blocks_vs_strips_predictions.csv` and the per-P PNGs. It uses the **corrected** Part 0 fit summary (included in this overlay). The fit is `t_msg(m_bytes) = t_s + m_bytes*t_w`. **Task 8.2 expresses message size in double values; convert `t_w,value = 8*t_w,byte`.**

The script shows the idealized *interior-rank* formulas from Task 8.2 AND a refined approximation that accounts for the actual process-grid boundaries. Its representative parameters are the **median of the three inter-VM** fit coefficients. This is a qualitative/simplified model: actual placement mixes intra-VM TCP and inter-VM Tailscale; different links have different costs and MPI may overlap exchanges. Predictions are **not** measurements.

Before running the mandated large experiment, test the measurement machinery with a small diagnostic:

```bash
python3 scripts/compare_decomps.py \
  --measure --hosts heat-mathieu:4,heat-naji:4,heat-chris:4 \
  --tailscale --Ns 512 --ranks 4 --steps 10 --repeats 1 \
  --out-dir results/part3/diagnostic
```

The requested **full experimental matrix** requires genuine MPI measurements, and may be exceptionally slow across your measured Tailscale links. Once resource/time availability is confirmed, use:

```bash
python3 scripts/compare_decomps.py \
  --measure --hosts heat-mathieu:4,heat-naji:4,heat-chris:4 \
  --tailscale --Ns 512,2048,8192 --ranks 4,8 \
  --steps 500 --repeats 3 --map-by node \
  --out-dir results/part3/performance_node
```

For Part 4, compare `--map-by node` and `--map-by slot` in **separate out directories**, so the results do not mix.

**Resource warning:** An N=8192 global field has 8193² doubles (approximately 0.5 GiB per global time level), and 500 time steps across slow inter-VM links can take very long. The measurement script intentionally omits `--out` to avoid an expensive global gather, preserves each completed raw row, and skips already-recorded runs when resumed. Test with smaller N/steps before large experiments; do not launch multiple performance runs concurrently.

For real measurements the script saves:

- `blocks_vs_strips_raw.csv` — every execution's `t_total`, `t_comp`, `t_comm` and repeat index.
- `blocks_vs_strips_measured_medians.csv` — medians, min/max spreads, actual faster decomposition by total runtime.
- `blocks_vs_strips_predictions.csv` — analytic estimates.
- `blocks_vs_strips_P4.png`, `blocks_vs_strips_P8.png` — model curves and observed communication medians.

**Timing caveat:** the reduced maximum `t_total`, `t_comp`, and `t_comm` may come from different ranks; they need not sum exactly.

## 7. Git, team sync, evidence and remaining acceptance

After local builds and tests pass on Mathieu:

```bash
cd /home/mpi/heat-mpi
git status
git add CMakeLists.txt README.md PART3_BLOCKS_GUIDE.md \
  src/heat.hpp src/heat_main.cpp src/heat_blocks.cpp \
  scripts/test_serial.py scripts/test_blocks.py scripts/compare_decomps.py \
  data/part0_corrected_fit_summary.csv llm_log/
git commit -m "Implement and test Part 3 MPI Cartesian block decomposition"
git push origin main
```

On **Naji's** and **Chris's** VMs:

```bash
cd /home/mpi/heat-mpi
git status
git pull --ff-only origin main
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/test_serial.py --exe build/heat
```

Their `test_blocks.py` local command is above. Save logs and CSVs under `results/part3/` and share them for the report. Check `.gitignore` before assuming evidence is tracked by Git.

**Do not declare Part 3 experimentally complete until genuine MPI tests pass on your Ubuntu VMs and the required performance measurements are collected.** The packaged source was locally tested using a separate file-based MPI protocol emulator because Open MPI was unavailable in the authoring environment; that emulator is **not included in the deliverable** and is **not a substitute for real MPI testing**.

### LLM logs

The professor requires the full actual prompts/responses, dates, model, team member, and changed files. The included `llm_log/PART3_REVIEW_CHECKLIST.md` is **only a tracking checklist**. Preserve/export the actual full interaction transcript separately and attach it to your project. Never commit credentials, SSH keys, or other secrets.

### Mixed-architecture note

If your cluster combines x86-64 and ARM64, the project requires documenting bit-for-bit equality on one architecture and reporting the maximum numerical difference on the mixed cluster. The block checker offers `--mixed-arch-report` to preserve mismatch rows and the largest absolute difference rather than stopping at the first mismatch. The regular tests **without** that switch still enforce exact byte equality. Cross-architecture differences are not automatically an acceptable substitute for correct algorithms; investigate and document them.
