#!/usr/bin/env python3
"""Resume an MPI ping-pong benchmark from an existing CSV without rerunning sizes.

Reads and validates a CSV written by mpi_pingpong, launches the same C++ MPI
executable with --start-bytes/--end-bytes, and appends one verified row per
finished message size. A second run can safely resume after Ctrl+C.

This is an MPI *launcher* script: run it on ONE VM only.
"""

import argparse
import csv
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

MIN_BYTES = 8
MAX_BYTES = 16 * 1024 * 1024
SIZES = [1 << exponent for exponent in range(3, 25)]
BASE_COLUMNS = ["bytes", "repetitions", "mean_rtt_s", "mean_oneway_s"]
V2_COLUMNS = BASE_COLUMNS + [
    "median_oneway_s", "min_oneway_s", "max_oneway_s", "batches"
]


def positive_integer(text):
    try:
        value = int(text)
        if value < 1:
            raise ValueError()
        return value
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc


def nonnegative_integer(text):
    try:
        value = int(text)
        if value < 0:
            raise ValueError()
        return value
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a nonnegative integer") from exc


def byte_size(text):
    value = positive_integer(text)
    if value not in SIZES:
        raise argparse.ArgumentTypeError(
            f"must be a power of two from {MIN_BYTES} to {MAX_BYTES}"
        )
    return value


def validate_data_row(row, columns, expected_bytes):
    if len(row) != len(columns):
        raise ValueError(
            f"expected {len(columns)} columns, found {len(row)} for size {expected_bytes}"
        )
    try:
        bytes_value = int(row[0])
        repetitions = int(row[1])
        if bytes_value != expected_bytes:
            raise ValueError(
                f"expected message size {expected_bytes}, found {bytes_value}"
            )
        if repetitions <= 0:
            raise ValueError("repetitions must be positive")
        for field in row[2:7] if columns == V2_COLUMNS else row[2:4]:
            value = float(field)
            if not math.isfinite(value) or value < 0:
                raise ValueError("timings must be finite and nonnegative")
        if columns == V2_COLUMNS and int(row[7]) <= 0:
            raise ValueError("batches must be positive")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid measurement row for {expected_bytes}: {exc}") from exc


def inspect_csv(path):
    """Return (header, number_of_finished_sizes). Never silently discard rows."""
    if not path.exists() or path.stat().st_size == 0:
        return V2_COLUMNS, 0
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except StopIteration:
            return V2_COLUMNS, 0
        if header not in (BASE_COLUMNS, V2_COLUMNS):
            raise ValueError(
                "CSV header does not match mpi_pingpong (supported: 4-column "
                "legacy CSV or 8-column v2 CSV). Original file was not changed."
            )
        count = 0
        for line, row in enumerate(reader, start=2):
            if count >= len(SIZES):
                raise ValueError(f"too many data rows (starting at CSV line {line})")
            try:
                validate_data_row(row, header, SIZES[count])
            except ValueError as exc:
                raise ValueError(f"CSV line {line}: {exc}") from exc
            count += 1
        return header, count


def make_command(args, start_bytes):
    hosts = [host.strip() for host in args.hosts.split(",")]
    if not (1 <= len(hosts) <= 2) or not all(hosts):
        raise ValueError("--hosts must contain one host or two comma-separated hosts")
    if len(hosts) == 2 and hosts[0] == hosts[1]:
        hosts = [hosts[0]]
    if args.transport == "shm" and len(hosts) != 1:
        raise ValueError("shared memory works only on one VM; use --transport tcp for two VMs")

    host_argument = f"{hosts[0]}:2" if len(hosts) == 1 else f"{hosts[0]}:1,{hosts[1]}:1"
    command = [
        args.mpirun, "--host", host_argument, "-np", "2", "--mca", "pml", "ob1"
    ]
    if args.transport == "tcp":
        command += [
            "--mca", "oob_tcp_if_include", "tailscale0",
            "--mca", "btl_tcp_if_include", "tailscale0",
            "--mca", "btl", "self,tcp",
        ]
    else:
        command += ["--mca", "btl", f"self,{args.shm_btl}"]

    command += [
        args.executable,
        "--batches", str(args.batches),
        "--warmups", str(args.warmups),
        "--start-bytes", str(start_bytes),
        "--end-bytes", str(args.end_bytes),
    ]
    return command


def stop_group(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run(args):
    path = args.csv.expanduser()
    header, done = inspect_csv(path)
    first_remaining = SIZES[done] if done < len(SIZES) else None
    if first_remaining is None:
        print(f"Already complete: all {len(SIZES)} message sizes are recorded in {path}")
        return 0
    if first_remaining > args.end_bytes:
        print(
            f"No sizes left up to --end-bytes={args.end_bytes}. "
            f"Next unfinished size is {first_remaining}."
        )
        return 0

    command = make_command(args, first_remaining)
    print(f"CSV: {path}", flush=True)
    print(f"Completed: {done}/{len(SIZES)} sizes", flush=True)
    print(f"Resuming at: {first_remaining} bytes", flush=True)
    print(f"Stopping at: {args.end_bytes} bytes", flush=True)
    print(f"Command: {shlex.join(command)}", flush=True)
    if args.dry_run:
        print("Dry run; no files changed, no MPI process launched.")
        return 0

    path.parent.mkdir(parents=True, exist_ok=True)
    # A completed last row may lack a trailing newline (e.g. from older tools).
    # Separate it before appending; don't delete any measurements.
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb+") as handle:
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) not in (b"\n", b"\r"):
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())

    proc = None
    appended = 0
    try:
        with path.open("a", newline="", encoding="utf-8") as output:
            writer = csv.writer(output, lineterminator="\n")
            if path.stat().st_size == 0:
                writer.writerow(header)
                output.flush()
                os.fsync(output.fileno())
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=None,  # MPI diagnostics visible to the user, not in the CSV.
                text=True,
                encoding="utf-8",
                bufsize=1,
                start_new_session=True,
            )
            assert proc.stdout is not None
            seen_header = False
            remaining_sizes = [size for size in SIZES[done:] if size <= args.end_bytes]
            for raw_line in proc.stdout:
                stripped = raw_line.rstrip("\r\n")
                if not stripped:
                    continue
                data = next(csv.reader([stripped], strict=True))
                if not seen_header:
                    if data != V2_COLUMNS:
                        raise ValueError(
                            f"Unexpected MPI output header: {stripped!r}. "
                            "Rebuild the updated C++ executable on both VMs."
                        )
                    seen_header = True
                    continue
                if appended >= len(remaining_sizes):
                    raise ValueError("MPI program produced extra data rows")
                required_size = remaining_sizes[appended]
                validate_data_row(data, V2_COLUMNS, required_size)
                # If the existing CSV has the legacy 4-column format, preserve it.
                writer.writerow(data if header == V2_COLUMNS else data[:4])
                output.flush()
                os.fsync(output.fileno())
                appended += 1
                print(
                    f"Saved {required_size:>9} bytes "
                    f"({done + appended}/{len(SIZES)} complete)",
                    flush=True,
                )

            ret = proc.wait()
            if ret != 0:
                raise RuntimeError(f"mpirun exited with status {ret}")
            if not seen_header or appended != len(remaining_sizes):
                raise RuntimeError(
                    f"MPI produced only {appended}/{len(remaining_sizes)} expected rows"
                )
    except KeyboardInterrupt:
        if proc is not None:
            stop_group(proc)
        print(f"\nInterrupted. Saved {appended} new complete rows. Run the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, csv.Error) as exc:
        if proc is not None:
            stop_group(proc)
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Previously completed rows were preserved. Fix the issue and rerun.", file=sys.stderr)
        return 1

    print(f"Done. Appended {appended} rows to {path}.")
    if done + appended == len(SIZES):
        print("Benchmark complete: 22/22 sizes recorded.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Existing or new benchmark CSV")
    parser.add_argument("--hosts", required=True, help="One VM or two VMs separated by a comma")
    parser.add_argument("--transport", choices=("tcp", "shm"), default="tcp")
    parser.add_argument("--shm-btl", choices=("vader", "sm"), default="vader", help="vader for Open MPI 4, sm for Open MPI 5")
    parser.add_argument("--mpirun", default="mpirun")
    parser.add_argument("--executable", default="/home/mpi/heat-mpi/build/mpi_pingpong")
    parser.add_argument("--batches", type=positive_integer, default=3)
    parser.add_argument("--warmups", type=nonnegative_integer, default=5)
    parser.add_argument("--end-bytes", type=byte_size, default=MAX_BYTES)
    parser.add_argument("--dry-run", action="store_true", help="Validate CSV and show command only")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
