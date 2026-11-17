# MPI Ping-Pong Benchmark — Corrected Run Commands (Open MPI 4 / Ubuntu 24.04)

These commands assume the executable exists at `/home/mpi/heat-mpi/build/mpi_pingpong`
on every VM and MPI/SSH/Tailscale are already configured. They are to be launched
as the `mpi` user. The MPI executable is run across ranks automatically.

## 1. Rebuild the new C++ benchmark on **every VM**

Replace `src/mpi_pingpong.cpp` in each VM's common checkout, then:

```bash
cd /home/mpi/heat-mpi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
mpirun --version
```

The standard Ubuntu 24.04 packages provide Open MPI 4.1.x with the `vader`
shared-memory BTL. If you use Open MPI 5 instead, use `sm` in place of `vader`.

## 2. Same-VM shared memory — run on that VM

Example for heat-alice:

```bash
cd /home/mpi/heat-mpi
mkdir -p results
set -o pipefail
mpirun --host heat-alice:2 -np 2 \
  --mca pml ob1 \
  --mca btl self,vader \
  /home/mpi/heat-mpi/build/mpi_pingpong --batches 3 \
  | tee results/ping_same_heat-alice_shm.csv
```

Repeat on heat-bob and heat-carol using their own names and filenames. The `ob1`
PML forces use of the specified shared-memory BTL, rather than another PML.

## 3. Same-VM TCP-only comparison (useful extra, not substitute for shared memory)

From heat-alice:

```bash
mpirun --host heat-alice:2 -np 2 \
  --mca pml ob1 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl self,tcp \
  /home/mpi/heat-mpi/build/mpi_pingpong --batches 3 \
  | tee results/ping_same_heat-alice_tcp.csv
```

## 4. Cross-VM TCP over Tailscale — all commands run from ONE launcher

Check `tailscale ping` and record whether peers are direct or DERP before each pair.

**Alice ↔ Bob:**

```bash
cd /home/mpi/heat-mpi
mkdir -p results
set -o pipefail
mpirun --host heat-alice:1,heat-bob:1 -np 2 \
  --mca pml ob1 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl self,tcp \
  /home/mpi/heat-mpi/build/mpi_pingpong --batches 3 \
  | tee results/ping_heat-alice_heat-bob.csv
```

**Alice ↔ Carol:**

```bash
mpirun --host heat-alice:1,heat-carol:1 -np 2 \
  --mca pml ob1 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl self,tcp \
  /home/mpi/heat-mpi/build/mpi_pingpong --batches 3 \
  | tee results/ping_heat-alice_heat-carol.csv
```

**Bob ↔ Carol:**

```bash
mpirun --host heat-bob:1,heat-carol:1 -np 2 \
  --mca pml ob1 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl self,tcp \
  /home/mpi/heat-mpi/build/mpi_pingpong --batches 3 \
  | tee results/ping_heat-bob_heat-carol.csv
```

## 5. Validate results and generate latency/bandwidth plot

```bash
wc -l results/ping_heat-alice_heat-bob.csv   # should be 23 (header + 22 sizes)
head -n 3 results/ping_heat-alice_heat-bob.csv
python3 scripts/fit_pingpong.py results/ping_heat-alice_heat-bob.csv
python3 scripts/fit_pingpong.py results/ping_same_heat-alice_shm.csv
```

The C++ output keeps the original columns
`bytes,repetitions,mean_rtt_s,mean_oneway_s`, and adds
`median_oneway_s,min_oneway_s,max_oneway_s,batches`. This is backward-compatible
with the original fitting script. `repetitions` is per batch. `--batches 3` is
the default; for quick smoke tests you can use `--batches 1`, but do not use
those abbreviated runs as the sole final evidence.

For final measurements, repeat **each full command at least 3 times** with unique
CSV filenames; record VM versions, connection type (direct/DERP), Wi-Fi/wired,
and date/time. The program does not guarantee that same-VM TCP stays on the
loopback interface; it explicitly selects the TCP BTL, with the local IP route
determined by the OS/network configuration.
