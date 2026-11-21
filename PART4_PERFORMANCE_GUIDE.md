# Part 4 — Reproducible Performance Study

**Specification:** `heat_mpi_project.pdf`, Part 4, §§8.1–8.3. **Mathematical basis:** our cumulative mathematical task solutions, especially Task 8.2. **Numerical reference:** verified serial, strips, and blocks from Parts 1–3. No stencil or solver source is changed in Part 4.

## Measurement contract

- Fixed `--steps 200`, `--r 0.2`, Case 2; **three independent runs** minimum; record median and min/max spread. The specification gives `--steps 500` as an *example*, not a mandatory value.
- On each VM, use at most four MPI slots, matching the 4 vCPUs. Plug laptops in, stop other heavy programs, and record Wi-Fi/Ethernet and Tailscale direct/DERP status during tests.
- Timings come from the solver's time-stepping loop only, excluding MPI startup, field initialization, and output. Saved wall duration separately includes launch overhead. All runs should be sequential.
- **Important transport:** matching Part 3 uses `--mca btl self,tcp` and Tailscale TCP selection; co-located ranks use TCP, not the SHM measured in Part 0. Do not mix experimental transports. An optional separate `--transport hybrid-vader` mode exists for Open MPI 4, but it is *not* part of the main dataset and must use a different output root.
- Do not treat predicted values or synthetic test data as observations. Part 3 raw results are imported **without recomputation**, only for matching hosts/placement/transport/steps/r; imported rows carry source `part3-verified`.
- `run_perf.py` **plans without executing** by default. Add `--execute` for actual measurements. It flushes each successful run to CSV immediately, checks duplicates, saves the exact command and raw stdout/stderr, and resumes based on the full experiment key. If interrupted, rerun the same command.
- **Safety:** `--timeout 9000` applies *per MPI run*, not per experiment series. Expensive N=8192 distributed tests can still take many hours.

## A. Install Part 4 (on Mathieu first)

From the local computer with `Heat_MPI_Part4_Overlay.zip` in Downloads:

```powershell
scp "$HOME\Downloads\Heat_MPI_Part4_Overlay.zip" mpi@heat-mathieu:/home/mpi/
```

**Mathieu VM**:

```bash
cd /home/mpi/heat-mpi
git status
unzip -o /home/mpi/Heat_MPI_Part4_Overlay.zip -d .
python3 -m py_compile scripts/run_perf.py scripts/analyze_perf.py scripts/model_perf.py scripts/capture_env.py
python3 scripts/test_part4.py
python3 scripts/test_serial.py --exe build/heat
git diff --stat
# Add the project's REQUIRED full AI conversation log in llm_log/ before commit.
git add PART4_PERFORMANCE_GUIDE.md scripts/run_perf.py scripts/analyze_perf.py scripts/model_perf.py scripts/capture_env.py scripts/test_part4.py llm_log/
git commit -m "Add Part 4 reproducible performance study"
git push origin main
```

The scripts use the existing `build/heat` from Part 3; C++ rebuild is unnecessary if that executable is present and current. If missing, build with `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2`.

**Naji VM and Chris VM, each independently:**

```bash
cd /home/mpi/heat-mpi
git status
git pull --ff-only origin main
python3 scripts/test_part4.py
test -x build/heat || { cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j2; }
```

If `git status` reports uncommitted work, save/commit/stash it first; don't blindly overwrite or discard teammates' changes.

## B. Record VM environment (each VM)

Choose actual connection type (`wifi`, `ethernet`, `other` or `unknown`) instead of copying the example blindly.

**Mathieu VM:**

```bash
cd /home/mpi/heat-mpi
python3 scripts/capture_env.py --vm mathieu --connection unknown
```

**Naji VM:**

```bash
cd /home/mpi/heat-mpi
python3 scripts/capture_env.py --vm naji --connection unknown
```

**Chris VM:**

```bash
cd /home/mpi/heat-mpi
python3 scripts/capture_env.py --vm chris --connection unknown
```

Check links before/during remote runs on Mathieu (write to separate files if you want evidence):

```bash
mkdir -p results/part4/metadata
tailscale ping heat-naji | tee results/part4/metadata/ping_naji.txt
tailscale ping heat-chris | tee results/part4/metadata/ping_chris.txt
```

A ping that becomes direct does not prove every timed benchmark remained direct. Note any DERP fallback or Wi-Fi changes.

## C. Serial baseline — execute on ALL three VMs separately

**Mathieu VM:**

```bash
cd /home/mpi/heat-mpi
mkdir -p results/part4/baseline/mathieu
set -o pipefail
python3 scripts/run_perf.py baseline --vm mathieu --execute \
  --Ns 512,1024,2048,4096,8192 --steps 200 --r 0.2 --repeats 3 \
  | tee results/part4/baseline/mathieu/run.log
wc -l results/part4/baseline/mathieu/raw.csv
```

**Naji VM:**

```bash
cd /home/mpi/heat-mpi
mkdir -p results/part4/baseline/naji
set -o pipefail
python3 scripts/run_perf.py baseline --vm naji --execute \
  --Ns 512,1024,2048,4096,8192 --steps 200 --r 0.2 --repeats 3 \
  | tee results/part4/baseline/naji/run.log
wc -l results/part4/baseline/naji/raw.csv
```

**Chris VM:**

```bash
cd /home/mpi/heat-mpi
mkdir -p results/part4/baseline/chris
set -o pipefail
python3 scripts/run_perf.py baseline --vm chris --execute \
  --Ns 512,1024,2048,4096,8192 --steps 200 --r 0.2 --repeats 3 \
  | tee results/part4/baseline/chris/run.log
wc -l results/part4/baseline/chris/raw.csv
```

Each baseline CSV should contain **16 lines = 15 runs + header**. Record the machine's connection type in metadata. Never label results from the wrong VM: the runner checks its hostname.

### Gather Naji and Chris baselines onto Mathieu (run on Mathieu)

```bash
cd /home/mpi/heat-mpi
mkdir -p results/part4/baseline/naji results/part4/baseline/chris
scp -r mpi@heat-naji:/home/mpi/heat-mpi/results/part4/baseline/naji/* results/part4/baseline/naji/
scp -r mpi@heat-chris:/home/mpi/heat-mpi/results/part4/baseline/chris/* results/part4/baseline/chris/
mkdir -p results/part4/metadata
scp mpi@heat-naji:/home/mpi/heat-mpi/results/part4/metadata/naji.json results/part4/metadata/
scp mpi@heat-chris:/home/mpi/heat-mpi/results/part4/metadata/chris.json results/part4/metadata/
```

## D. Reuse the already-validated Part 3 strong-scaling measurements (Mathieu only)

```bash
cd /home/mpi/heat-mpi
python3 scripts/run_perf.py import-part3 \
  --csv results/part3/performance_node_200/blocks_vs_strips_raw.csv \
  --hosts heat-mathieu:4,heat-naji:4,heat-chris:4
wc -l results/part4/strong/raw.csv
```

Expected on first import: **36 rows imported / 37 CSV lines**. Repeated imports skip the same data. It rejects incomplete, inconsistent or incorrectly configured source files.

## E. Strong scaling — Mathieu is the launcher

Measure both decompositions, **N = 512,2048,8192**, **P = 1 ... 12**, and placements **node** and **slot**. Use your validated 200-step data; no need to re-run node/P4/P8.

Set this once in the same terminal:

```bash
cd /home/mpi/heat-mpi
export HOSTS='heat-mathieu:4,heat-naji:4,heat-chris:4'
mkdir -p results/part4/strong
set -o pipefail
```

### E0. New P=12 correctness smoke test (Mathieu)

Parts 2–3 were validated on your cluster for ranks up to P=8. **Before measuring P=12, verify correctness with all twelve slots**:

```bash
cd /home/mpi/heat-mpi
export HOSTS='heat-mathieu:4,heat-naji:4,heat-chris:4'
mkdir -p results/part4/smoke
./build/heat --decomp serial --case 2 --N 32 --r 0.2 --steps 9 \
  --out results/part4/smoke/serial.bin
for decomp in strips blocks; do
  mpirun --host "$HOSTS" --map-by node -np 12 \
    --mca pml ob1 --mca oob_tcp_if_include tailscale0 \
    --mca btl_tcp_if_include tailscale0 --mca btl self,tcp \
    /home/mpi/heat-mpi/build/heat --decomp "$decomp" --case 2 \
    --N 32 --r 0.2 --steps 9 \
    --out "/home/mpi/heat-mpi/results/part4/smoke/${decomp}_p12.bin"
  cmp results/part4/smoke/serial.bin "results/part4/smoke/${decomp}_p12.bin" \
    && echo "PASS: P=12 ${decomp} equals serial"
done
```

Both comparisons must pass before using P=12 for performance data. This is a new process count beyond the original Part 3 P<=8 acceptance matrix.

### E1. First informative process counts, `--map-by node`

Run one grid size at a time, in increasing order. You can stop between iterations; all successful repeats stay in the CSV.

```bash
for N in 512 2048 8192; do
  python3 scripts/run_perf.py strong --execute \
    --hosts "$HOSTS" --Ns "$N" --Ps 1,2,3,4,6,8,12 \
    --placement node --steps 200 --r 0.2 --repeats 3 --timeout 9000 \
    | tee "results/part4/strong/node_N${N}_stage1.log"
done
```

### E2. Same informative process counts, `--map-by slot`

```bash
for N in 512 2048 8192; do
  python3 scripts/run_perf.py strong --execute \
    --hosts "$HOSTS" --Ns "$N" --Ps 1,2,3,4,6,8,12 \
    --placement slot --steps 200 --r 0.2 --repeats 3 --timeout 9000 \
    | tee "results/part4/strong/slot_N${N}_stage1.log"
done
```

**CHECKPOINT:** review the data and timings before launching additional P values. Your Tailscale links are extremely slow; in particular P=8,N=8192 took a long time in Part 3. The first stage intentionally produces a useful partial scaling comparison even if resource/time constraints prevent the exhaustive sweep.

### E3. Complete the literal P=1,...,12 sweep (remaining ranks)

The professor says P from 1 up to available slots; to cover **every integer P**, run these remaining configurations:

```bash
for placement in node slot; do
  for N in 512 2048 8192; do
    python3 scripts/run_perf.py strong --execute \
      --hosts "$HOSTS" --Ns "$N" --Ps 5,7,9,10,11 \
      --placement "$placement" --steps 200 --r 0.2 --repeats 3 --timeout 9000 \
      | tee "results/part4/strong/${placement}_N${N}_stage2.log"
  done
done
```

After E1–E3, full strong scaling contains **144 distinct configurations x 3 = 432 runs**, of which 36 were imported from Part 3. The full CSV therefore has **433 lines**, when performed under exactly these settings. Note: 432 runs over a slow Tailscale cluster is exceptionally expensive; run progressively and do not launch the whole matrix without checking costs.

### E4. Record actual MPI placement (Mathieu)

Our model infers mapping from `--map-by node`/`slot`. Verify rank placement and keep it with the results:

```bash
cd /home/mpi/heat-mpi
export HOSTS='heat-mathieu:4,heat-naji:4,heat-chris:4'
mkdir -p results/part4/metadata
for placement in node slot; do
  for P in 4 8 12; do
    mpirun --host "$HOSTS" --map-by "$placement" -np "$P" \
      --mca pml ob1 --mca oob_tcp_if_include tailscale0 \
      --mca btl_tcp_if_include tailscale0 --mca btl self,tcp \
      bash -c 'echo "rank=$OMPI_COMM_WORLD_RANK host=$(hostname)"' \
      | sort -V | tee "results/part4/metadata/placement_${placement}_P${P}.txt"
  done
done
```

If observed rank assignment differs from the model's inferred order, revise the model's mapping before claiming quantitative agreement. The measurements themselves remain valid.

## F. Weak scaling — Mathieu launches across the cluster

Use approximately `1024 x 1024` interior nodes per rank. The runner computes N from `round(1 + 1024*sqrt(P))` and writes the actual N/P into CSV. This keeps per-rank work approximately constant, not exactly equal because N is integer and strips/blocks may distribute unevenly.

**Mathieu VM:**

```bash
cd /home/mpi/heat-mpi
export HOSTS='heat-mathieu:4,heat-naji:4,heat-chris:4'
mkdir -p results/part4/weak
set -o pipefail
for P in 1 2 3 4 6 8 12; do
  python3 scripts/run_perf.py weak --execute \
    --hosts "$HOSTS" --Ps "$P" --base-side 1024 \
    --placement node --steps 200 --r 0.2 --repeats 3 --timeout 9000 \
    | tee "results/part4/weak/node_P${P}.log"
done
```

**Expected weak CSV:** 14 configurations (2 decompositions x 7 rank counts), each repeated 3 times = **42 measurements + header**, so **43 lines**. Node placement is the primary weak-scaling study; repeat with `--placement slot` only if additional comparison is desirable (use the same CSV, settings become a distinct key).

## G. Analyze results and fit the mathematical performance model (Mathieu only)

Can be run even while data are partial; it explicitly states coverage and does not call incomplete groups *complete*.

```bash
cd /home/mpi/heat-mpi
mkdir -p results/part4
python3 scripts/analyze_perf.py --out-root results/part4 \
  | tee results/part4/analysis.log
python3 scripts/model_perf.py --out-root results/part4 \
  --fit-summary data/part0_corrected_fit_summary.csv \
  | tee results/part4/model.log
cat results/part4/analysis/coverage.txt
ls -lh results/part4/figures/
```

**Analysis outputs:** `analysis/baseline_summary.csv`, `strong_summary.csv`, `weak_summary.csv`, `coverage.txt`. Figures: serial times/cost, strong runtime/speedup/efficiency/communication share by N, weak scaling, model-versus-measured by N, and a `t_s=1us` counterfactual.

**Model inputs/assumptions:** Task 8.2, per-VM compute seconds per node update from serial, exact row/block sizes, measured pairwise inter-VM startup latency and seconds/byte, and critical rank's estimated computation+halo cost. Communication constants in the math PDF count doubles, so 8 bytes/double must be applied. The model is approximate; the **same-VM TCP** cost is unmeasured and **modeled as zero** by default (thus only a lower bound on local halo overhead). `model/model_assumptions.txt` states this prominently. If you independently benchmark TCP loopback on the VMs, you can provide appropriate parameters using `--local-ts-us` and `--local-tw-s-per-byte`; do **not** pass the Part 0 SHM parameters instead. The `t_s=1us` scenario modifies only cross-VM latency, retaining observed bandwidth, so it is not a realistic full HPC upgrade forecast.

**How to interpret measured communication share:** `median(max_rank_comm) / median(max_rank_total)` (maxima reduced separately). This may not be the actual fraction on any single rank; it is a reproducible proxy based on existing solver output.

## H. Final evidence checks (Mathieu)

```bash
cd /home/mpi/heat-mpi
wc -l results/part4/baseline/{mathieu,naji,chris}/raw.csv
wc -l results/part4/strong/raw.csv
wc -l results/part4/weak/raw.csv
cat results/part4/analysis/coverage.txt
find results/part4/figures -maxdepth 1 -type f -name '*.png' | sort
```

Verify no duplicate/rejected rows; do not treat the result folder as a proof of completion by itself. Retain individual run logs, system metadata, raw CSVs, the Part 0 network CSVs, the Part 1 convergence/instability PNGs, and the Part 2/3 correctness and Part 3 model comparison evidence. All 48 Part 3 three-way comparisons already passed in `results_v3.rar`.

Git: `results/` may be ignored by `.gitignore`. Keep a backed-up archive and selectively force-add the relevant results or stage them using `git add -f ...` after checking file sizes. Avoid committing huge `.bin` outputs or private Tailscale peer details. **Maintain the complete professor-required LLM interaction logs** (prompts, entire responses, dates, models, team members, modified files); the distributed template file is only a checklist, not the actual transcript.

## I. Reporting questions (Part 4, §8.3)

1. For which N does spreading across machines pay off? Use strong-scaling runtime versus Mathieu serial (and optionally fastest serial VM), not just whether blocks beat strips.
2. Does the model predict the same crossover? Compare `model_vs_measured.csv` and the plotted curves; where it disagrees explain network link heterogeneity, possible DERP, Wi-Fi, TCP loopback, synchronization, cache and slower VM rates.
3. What would change with t_s~1 microsecond? Use the separate scenario PNGs **with bandwidth held constant** and explicitly note the assumption.
4. What advice for Tailscale users? Use data on communication share and scaling, not speculation; discuss fewer inter-VM boundaries, slot-vs-node placement, and when parallelism is not worthwhile.

No solver code changes are needed to complete these measurements. The real MPI benchmarks must be performed on the VMs; local tests here exercise the Python pipeline only.
