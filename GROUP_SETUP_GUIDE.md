# Parallel 2D Heat Diffusion with MPI over Tailscale
## Group setup manual — repository, cluster, and network measurements

**Project phase:** Part 0, plus the repository/LLM logs needed throughout Parts 1–4.  
**Source of course requirements:** *heat_mpi_project.pdf*, especially Sections 4, 10–12.  
**What this starter does not do:** implement the serial or parallel heat solver; those are the next phase.  
**Execution rule:** commands marked **HOST** run on the real laptop, commands marked **VM** run inside Ubuntu; commands marked **ALL VMs** must be performed by every teammate on their own VM.

## Team-wide agreements — make these decisions before running anything

1. Every teammate will run an **Ubuntu 24.04 LTS** VM, with **at least 2 vCPUs and 4 GiB RAM**; allocate 20–25 GiB disk or more. Multipass is the simplest cross-platform choice, but VirtualBox, VMware, Hyper-V or UTM are permitted by the course.
2. Use the exact same Linux username everywhere: **`mpi`**. Each teammate may choose a different password; don't share passwords with the group. The project checkout path will be **`/home/mpi/heat-mpi`** on every VM.
3. Pick one unique VM/Tailscale hostname per person, such as `heat-alice`, `heat-bob`, `heat-carol` (replace these names in every command). Use lowercase letters/hyphens, no spaces.
4. Appoint **one tailnet administrator** to create the Tailscale tailnet and invite the others. Invite only teammates; do not make the network public.
5. Choose a **private shared Git repository** (e.g. on GitHub) that all teammates can clone and push to. Assign one person to maintain the hostfile and measurement summaries.
6. Make a shared machine inventory with columns: member, host OS, host CPU, Ubuntu VM hostname, CPU architecture (`uname -m`), VM vCPUs, VM RAM, physical network (Wi-Fi/Ethernet), Open MPI version, Tailscale path for each peer (direct/DERP).
7. Start archiving **all complete AI/LLM interactions now**. The professor requires complete prompts and responses, even discarded ones, including this project's earlier mathematics conversation. Preserve complete transcripts; summaries alone are not sufficient. Protect secrets and private keys.

Suggested assignments: one person coordinates the repo and LLM logs; each person builds their own VM, Tailscale membership, and SSH; one person coordinates the MPI ping-pong matrix and aggregates results. **Everyone still performs the local steps on their own VM.**

---

# STEP 1 — Git repository and mandatory LLM logs

## 1A. Create the private shared repository

**Repository coordinator** creates an empty private remote repository named `heat-mpi`, gives teammates access, and commits the contents of this starter. Make the entire source/guide available to teammates through Git; do not send compiled executables across mixed CPU architectures.

After copying the starter files into an empty working directory (HOST or VM), initialize Git:

```bash
git init -b main
git add .
git commit -m "Bootstrap MPI cluster tools and LLM logging"
git remote add origin YOUR_PRIVATE_REPOSITORY_URL
git push -u origin main
```

Replace `YOUR_PRIVATE_REPOSITORY_URL` with your actual remote repository URL. Configure your own `git config --global user.name` and `git config --global user.email` first if Git asks.

Alternatively, create the repository directly on the remote site, upload the starter files, and clone it on each VM; **do not re-initialize an existing clone**.

**Every teammate, on their Ubuntu VM as user `mpi`:**

```bash
cd /home/mpi
git clone YOUR_PRIVATE_REPOSITORY_URL heat-mpi
cd /home/mpi/heat-mpi
ls -la
```

Expected: `AGENTS.md`, `CMakeLists.txt`, `GROUP_SETUP_GUIDE.md`, `hosts.example`, `src/`, `scripts/`, and `llm_log/` are present. If the Git remote requires authentication, use your own account's secure Git credential/SSH mechanism. Do not share access tokens or add them to LLM logs.

If you already created a repository, **merge these starter files into it** rather than replacing teammates' work.

## 1B. Make `llm_log/` reliable from day one

The professor requires a full record of **every** LLM interaction used for cluster setup, coding, debugging, data analysis and report writing, including rejected answers. Each entry/session must state:

- Date and time with timezone.
- Team member.
- Tool/service and model.
- **Full original prompt and full response (verbatim)**, or a saved raw transcript containing them.
- Every file created/modified (with line ranges when practical).
- Related commit hash once available.

`AGENTS.md` in this starter tells compatible coding agents to do this, but **an instruction file is not an automatic logging guarantee**. Check each session yourself, and export/copy complete transcripts for chat-based tools.

Suggested folder organization:

```text
llm_log/
  README.md
  2026-11-15_member-a_chatgpt_setup.md
  2026-11-15_member-b_coding-agent_mpi.md
  raw_exports/
```

Log the earlier mathematical discussion too if used to produce the project deliverable; do not replace it with a summary. For this setup session, save this full conversation response or the exported transcript. After logging, commit logs with the related project changes:

```bash
git add AGENTS.md llm_log/ src/ scripts/ CMakeLists.txt GROUP_SETUP_GUIDE.md
git commit -m "Record setup work and full LLM interaction logs"
git push
```

**Security:** Never commit `.ssh` private keys, Tailscale auth keys, passwords, personal login tokens, `.env` secrets or raw private information not needed for the coursework. Keep the repository private.

---

# STEP 2 — One Ubuntu VM and a connected MPI cluster

## 2A. Install a virtual machine manager (HOST, each teammate)

The course allows any hypervisor, but these instructions use **Canonical Multipass** on Windows, macOS or Linux. Install it from the official Multipass website/documentation, enable CPU virtualization if required, and open PowerShell/Terminal.

Create your VM, replacing `heat-alice` with your assigned hostname:

```bash
multipass launch 24.04 --name heat-alice --cpus 2 --memory 4G --disk 25G
multipass list
multipass info heat-alice
multipass shell heat-alice
```

`multipass launch` and `multipass shell` are **HOST commands**; everything after entering the shell is executed **inside the Ubuntu VM** until you type `exit`.

If your computer has plenty of resources, assigning 4 vCPUs and 6–8 GiB RAM is fine. Do not allocate more than your host can comfortably spare. The professor's minimum is 2 vCPUs and 4 GiB RAM. A large disk will help for later benchmark outputs.

If you use another hypervisor, create an **Ubuntu Server 24.04 LTS** VM with the same resources and continue at Step 2B. Default VM NAT networking is acceptable because Tailscale will run **inside** the VM.

## 2B. Verify Ubuntu and install course software (VM, ALL VMs)

```bash
cat /etc/os-release
uname -m
nproc
free -h
hostname
sudo apt update
sudo apt install -y build-essential cmake git openmpi-bin libopenmpi-dev \
  openssh-server python3-numpy python3-matplotlib curl
sudo systemctl enable --now ssh
mpirun --version
mpic++ --version
cmake --version
```

Check that `/etc/os-release` reports **24.04**, `nproc` is at least 2, and `mpirun` runs. **All VMs must use the same Ubuntu release and matching Open MPI version.** Record output in the machine inventory. `uname -m` may be `x86_64` on Intel/AMD or `aarch64` on Apple Silicon; note mixed architectures in the report.

## 2C. Create the common Linux user (VM, ALL VMs)

From the initial Ubuntu account:

```bash
sudo adduser mpi
sudo usermod -aG sudo mpi
sudo -iu mpi
whoami
pwd
```

Choose a local password when `adduser` asks. `whoami` must show `mpi`, and `pwd` should show `/home/mpi`. From here on, do the ordinary MPI and Git commands as `mpi`, **not as root**. For system administration, return to the initial user or run an authorized `sudo` command. If user `mpi` already exists, don't recreate it.

## 2D. Join the team's Tailscale network (VM, ALL VMs)

The **tailnet administrator** creates a tailnet and invites each teammate through the Tailscale admin console. Each teammate accepts that invitation using their own login.

On the VM (either the initial account or `mpi`, with sudo rights):

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=heat-alice
tailscale status
tailscale ip -4
ip -br address show tailscale0
```

Open the authentication URL printed by `tailscale up` in a browser and log in to the **team's** tailnet (not a separate unrelated tailnet). Run `tailscale status` again. The interface must be named `tailscale0`, and the device should appear in the same Tailscale admin console as the others.

The official alternative to `curl | sh` is the manual package-install method at Tailscale's stable packages website.

**Important:** Tailscale must run inside every Ubuntu VM, not just on the Windows/Mac host.

After two or more VMs join, test connectivity **from each VM to all others**:

```bash
ping -c 3 heat-bob
tailscale ping heat-bob
tailscale status
```

Use each real peer hostname. Tailscale ping reports whether the peer is reached by **direct** connection or via **DERP relay**. A relay is allowed but can be much slower. Record every pair's result; don't assume all machines connect directly. Re-run Tailscale ping after it has had a chance to establish a direct path.

## 2E. Set up passwordless SSH in both directions (VM, ALL VMs)

`mpirun` launches remote processes over SSH. The team needs **every-to-every** passwordless SSH, not just SSH from the group leader's VM.

On every VM, **as `mpi`**:

```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519
ls -l ~/.ssh/id_ed25519.pub
```

If you already have a key at that path, **do not overwrite it**; use the existing key or create a new specifically named project key. The project example uses a key without a passphrase for unattended MPI launches. Protect the private key by permissions and never copy/share the private key.

For each remote VM, run (example from `heat-alice`):

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub mpi@heat-bob
ssh-copy-id -i ~/.ssh/id_ed25519.pub mpi@heat-carol
ssh mpi@heat-bob hostname
ssh mpi@heat-carol hostname
```

On the first copy, the target `mpi` user's password is expected. The remote target must already have Tailscale and the SSH server running.

**Repeat SSH setup from `heat-bob` to the other hosts, from `heat-carol` to the others, etc.** Each user owns their own private SSH key. Test noninteractively:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 mpi@heat-bob hostname
```

Expected: prints `heat-bob` (or the remote VM's configured Linux hostname), without a password, passphrase or host-key confirmation prompt. Accept a peer's host key on first interactive connection and retry `BatchMode` after that. Check fingerprints if the host-key identity is uncertain.

If SSH cannot reach port 22 but `tailscale ping` works, check `sudo systemctl status ssh`, the tailnet access policy, and the VM firewall. If `ufw` is active, a targeted rule is preferable to opening port 22 on the public Internet:

```bash
sudo ufw status
# Only if UFW is enabled and blocking SSH on the tailnet:
sudo ufw allow in on tailscale0 to any port 22 proto tcp
```

**Do not** configure public router port forwarding for this project.

## 2F. Clone and build on EVERY VM, at the same absolute path

As `mpi`:

```bash
cd /home/mpi
git clone YOUR_PRIVATE_REPOSITORY_URL heat-mpi
cd /home/mpi/heat-mpi
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ls -l build/mpi_hello build/mpi_pingpong
```

If the directory was already cloned in Step 1, run `git pull` rather than cloning it again.

This must be repeated on every VM. The binary path must be the same everywhere: `/home/mpi/heat-mpi/build/mpi_hello`, etc. On mixed x86/ARM machines, build **locally on each VM from the same source**, using `-ffp-contract=off`, rather than copying executables.

## 2G. Run the first multi-machine MPI job

Create `hosts` on the **launcher VM** (for example, `heat-alice`):

```bash
cd /home/mpi/heat-mpi
cp hosts.example hosts
nano hosts
```

Example `hosts` file, using your real Tailscale hostnames:

```text
heat-alice slots=2
heat-bob slots=2
heat-carol slots=2
```

`slots` should not exceed the VM's vCPU allocation for benchmark runs. From the launcher VM:

```bash
mpirun --hostfile hosts -np 6 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl tcp,self \
  hostname
```

Expected: six lines of hostnames, with two processes running on each of the three example VMs (output order may vary).

Then run the actual hello program:

```bash
mpirun --hostfile hosts -np 6 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl tcp,self \
  /home/mpi/heat-mpi/build/mpi_hello
```

Expected: output identifying ranks 0–5 and their hosts. If you see only the launcher VM's hostname, MPI is not using the intended remote hosts. The `tailscale0` filters matter because otherwise Open MPI may pick a private NAT interface unreachable from other VMs.

This invocation follows the Ubuntu 24.04/Open MPI setup required by the course. If your MPI release differs, check its actual available MCA parameters before changing commands.

---

# STEP 3 — Measure latency and bandwidth (Part 0, Section 4.7)

## 3A. What we must measure

Run an MPI **ping-pong** with exactly two ranks: rank 0 sends `m` bytes, rank 1 immediately sends the bytes back. Rank 0 measures elapsed round-trip time over repeated exchanges.

- Test payload sizes `m = 8, 16, 32, ..., 16,777,216` bytes (powers of two through 16 MiB).
- Measure at least one **same-VM** pair of ranks.
- Measure **every distinct cross-VM pair**. For 3 VMs: Alice–Bob, Alice–Carol, Bob–Carol.
- Use the same build and MPI networking settings for comparable measurements.
- Save raw CSV and note direct versus DERP, Wi-Fi versus Ethernet, VM vCPU/RAM, and any anomalies.
- Repeat each size; the provided program automatically performs warmups and multiple timed round trips.

Our provided `src/mpi_pingpong.cpp` reports:

```text
bytes,repetitions,mean_rtt_s,mean_oneway_s
```

The one-way estimate is:

\[
t_{oneway}(m)\approx \frac{t_{RTT}(m)}{2}.
\]

This assumes approximately symmetric request/response costs. It is an approximation, not a direct one-way clock-synchronized measurement.

## 3B. Same-VM tests: shared memory **and** TCP

Run each test while logged in as `mpi` on that VM (e.g. `heat-alice`). All machines
should use the same checked-out source and rebuilt program.

**Check the MPI version and available BTL component:**

```bash
mpirun --version
ompi_info --param btl vader --level 1 | head
```

Ubuntu 24.04 normally packages Open MPI 4.1.6, which uses `vader` for shared
memory. Open MPI 5 renames that BTL to `sm`; replace `vader` with `sm` if
needed. `--mca pml ob1` ensures that the selected BTLs are used for MPI messages.

**Test 1 — Shared-memory baseline (the primary same-VM baseline):**

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

**Test 2 — TCP-only on the same VM (optional, useful comparison):**

```bash
mpirun --host heat-alice:2 -np 2 \
  --mca pml ob1 \
  --mca oob_tcp_if_include tailscale0 \
  --mca btl_tcp_if_include tailscale0 \
  --mca btl self,tcp \
  /home/mpi/heat-mpi/build/mpi_pingpong --batches 3 \
  | tee results/ping_same_heat-alice_tcp.csv
```

The second test intentionally selects TCP for MPI messages; it does **not**
measure optimized same-VM shared memory. The exact local IP route depends on
operating-system networking. The shared-memory test excludes the TCP BTL for
application messages. MPI runtime/control traffic can still use TCP.

Run both tests on each VM if practical, replacing `heat-alice` in the command
and filename with that VM's hostname.

## 3C. Cross-VM tests

Run these from a **single MPI launcher** that has passwordless SSH to all
participants. First ensure the executable exists at the same absolute path
on each target VM. Choose exactly one rank on each VM:

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

Repeat with `heat-alice:1,heat-carol:1` and
`heat-bob:1,heat-carol:1`, saving to distinctly named CSVs.

Before each pair, record:

```bash
tailscale ping heat-bob
tailscale status
```

For every result, verify that the CSV has a header and 22 data rows:

```bash
wc -l results/ping_heat-alice_heat-bob.csv       # expect 23
head -n 2 results/ping_heat-alice_heat-bob.csv
```

`set -o pipefail` ensures a failed `mpirun` does not look successful just
because `tee` succeeded. Never interpret a partial CSV as a completed test.

The new program preserves `bytes,repetitions,mean_rtt_s,mean_oneway_s` and
appends `median_oneway_s,min_oneway_s,max_oneway_s,batches`.
`repetitions` means round trips **per batch**. For more reliable final results,
repeat the entire command at least three times with different CSV filenames;
record the conditions and the spread. Warm-ups default to 5 per message size.

## 3D. Fit the communication model

The course specifies the one-way model

\[
t_{msg}(m)=t_s+mt_w,
\]

where `m` is **bytes**, `t_s` is fixed message startup time in seconds, and `t_w` is seconds per byte. The effective bandwidth is

\[
B\approx 1/t_w\quad\text{(bytes/second)}.
\]

The accompanying `scripts/fit_pingpong.py` calculates:

- `t_s` as the median observed time for messages up to 64 bytes;
- `t_w` as a fixed-intercept least-squares fit over sizes at least 64 KiB;
- effective bandwidth in MiB/second;
- a log-log plot of measured one-way time and the model prediction.

This is a pragmatic two-regime estimate: measure startup cost using small messages and bandwidth using large messages, as the professor suggests. The straight-line model is approximate and can disagree near protocol thresholds.

On the VM that has the gathered CSVs:

```bash
cd /home/mpi/heat-mpi
python3 scripts/fit_pingpong.py results/ping_same_heat-alice_shm.csv
python3 scripts/fit_pingpong.py results/ping_heat-alice_heat-bob.csv
```

Repeat for every pair. The script saves a `.png` next to each `.csv`. When using `t_w` with the later heat-solver communication model, remember that a `double` is 8 bytes: sending `n` doubles corresponds to `m=8n` bytes.

## 3E. Compile one shared results table

Use this table format; fill with real measurements only:

| Rank placement / VM pair | Same VM or cross VM | Direct or DERP | Wi-Fi / Ethernet | `t_s` (microseconds) | Effective bandwidth (MiB/s) |
|---|---|---|---|---:|---:|
| heat-alice × 2 ranks | Same VM | N/A | Host network noted | measured | measured |
| heat-alice ↔ heat-bob | Cross VM | measured | measured | measured | measured |
| heat-alice ↔ heat-carol | Cross VM | measured | measured | measured | measured |
| heat-bob ↔ heat-carol | Cross VM | measured | measured | measured | measured |

For reproducibility, also record test date/time, CPU architecture, VM allocations, Open MPI version, message sizes, repetitions, any missing points, and script/git version used. Put all CSVs and charts into the repository; avoid committing unrelated huge files.

---

# TROUBLESHOOTING — check these before asking the group to rebuild

| Symptom | Likely cause / first action |
|---|---|
| `multipass` fails to launch | Check host virtualization/Hyper-V support, host RAM/disk, Multipass daemon/service. |
| Ubuntu isn't 24.04 | Recreate with `multipass launch 24.04 ...`; avoid a different default release. |
| `tailscale status` missing teammates | Confirm every VM joined the **same** tailnet and the admin invitation was accepted. |
| `tailscale ping` says DERP | It can still work. Record relay status; inspect firewalls/NAT and try again for a direct path. |
| SSH asks for a password | Ensure the public key was installed for **target user `mpi`**; check `~/.ssh/authorized_keys` and SSH permissions. |
| SSH asks to accept a host key | Connect interactively once, verify fingerprint, then rerun noninteractively. |
| `mpirun` hangs on multiple VMs | Check every-to-every SSH, Tailscale peer reachability, `tailscale0` MCA flags, same MPI versions and executable paths. |
| `mpirun` prints a missing executable | Build the project on **each VM** at `/home/mpi/heat-mpi` with `cmake --build build -j2`. |
| Cross-VM bandwidth extremely low | Check direct vs DERP, Wi-Fi quality, laptop power, VM CPU contention; don't fabricate corrected numbers. |
| MPI mismatch / unexpected behavior | Compare `mpirun --version` and `mpic++` on **all VMs**; update packages consistently. |
| Git accidentally contains credentials | Remove them immediately, rotate exposed secrets, clean repository history if needed; keep tailnet private. |

**Caution about `AGENTS.md`:** It is not a substitute for full export of chat sessions, especially conversations outside a coding agent. Check `llm_log/` after every work session.

---

# FINAL PART 0 / C1 ACCEPTANCE CHECKLIST

- [ ] A private shared Git repository exists, with starter files and a functioning build.
- [ ] `AGENTS.md` or `CLAUDE.md` and `llm_log/` are committed; all relevant complete AI transcripts are preserved.
- [ ] Every member has an Ubuntu **24.04** VM with >=2 vCPU and >=4 GiB RAM.
- [ ] Every VM has the same Linux username (`mpi`) and matches the agreed checkout path.
- [ ] Every VM reports a matching Open MPI version and has built the MPI programs locally.
- [ ] Every VM joins the **same** private Tailscale tailnet and can reach all other VMs by hostname.
- [ ] Tailscale direct/DERP status is recorded for every pair.
- [ ] Passwordless SSH succeeds in **both directions for every pair** with `BatchMode=yes`.
- [ ] From the launcher, `mpirun ... hostname` actually runs on all VMs.
- [ ] The MPI hello program prints all ranks with the expected host distribution.
- [ ] Ping-pong CSV exists for same-VM and **every cross-VM pair**, sizes 8 B–16 MiB.
- [ ] One-way times, latency, effective bandwidth and fit plots are saved.
- [ ] The machine inventory and network table are complete, with no invented numbers.
- [ ] Results and transcripts are committed; each member can explain the setup and code.

**Next after Part 0:** Build the serial C++ heat solver (Part 1), then verify it with Test V and the convergence study. The course's C1 checkpoint expects the cluster/network evidence **and** a passing serial solver, not only Part 0.

---

## Official documentation used for software-specific setup

- Ubuntu/Canonical Multipass (create Ubuntu 24.04 instance): https://documentation.ubuntu.com/canonical-kubernetes/release-1.34/snap/howto/install/multipass/
- Tailscale install on Linux: https://tailscale.com/docs/install/linux
- Tailscale direct vs DERP troubleshooting: https://tailscale.com/docs/reference/troubleshooting/network-configuration/derp-routing
- Open MPI TCP interface selection (releases corresponding to your Ubuntu package): https://www.open-mpi.org/faq/?category=tcp

The course's project PDF remains authoritative for assignment-specific requirements. Verify any VM/hypervisor variation against the course before substituting tools or workflows.
