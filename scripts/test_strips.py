#!/usr/bin/env python3
"""Part 2 checkpoint C2: compare MPI-strip output byte-for-byte with serial.

Run locally first, then on at least two VMs using --hosts or --hostfile.
Every MPI rank must find the executable at the same absolute path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', default='build/heat', help='Built heat executable, same absolute path on all VMs')
    p.add_argument('--mpi', default='mpirun', help='MPI launcher executable')
    p.add_argument('--hosts', help='Open MPI --host list, e.g. heat-mathieu:2,heat-naji:2')
    p.add_argument('--hostfile', help='Open MPI hostfile with slots (mutually exclusive with --hosts)')
    p.add_argument('--tailscale', action='store_true', help='Force TCP communication via tailscale0')
    p.add_argument('--ranks', default='1,2,3,4', help='Comma-separated rank counts, default 1,2,3,4')
    p.add_argument('--quick', action='store_true', help='Smallest C2 smoke matrix for cross-VM testing')
    p.add_argument('--timeout', type=int, default=240, help='Timeout (seconds) per MPI invocation')
    p.add_argument('--mixed-arch-report', action='store_true',
                   help='If x86/ARM ranks disagree, report largest absolute difference and continue')
    p.add_argument('--report', default='results/part2/strip_correctness.csv', help='CSV file for evidence')
    return p


def cmd_output(command, timeout):
    try:
        process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'TIMEOUT {timeout}s: {" ".join(command)}') from error
    if process.returncode != 0:
        raise RuntimeError(f'Command failed ({process.returncode}): {" ".join(command)}\n'
                           f'OUT:\n{process.stdout}\nERR:\n{process.stderr}')
    # The solver prints exactly one rank-0 key=value line (MPI may print warnings on stderr).
    lines = [line.strip() for line in process.stdout.splitlines() if line.startswith('decomp=')]
    if len(lines) != 1:
        raise RuntimeError(f'Expected exactly 1 heat summary line; got {lines!r}\n{process.stderr}')
    return dict(item.split('=', 1) for item in lines[0].split())


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def max_absolute_error(serial_bytes, strips_bytes):
    if len(serial_bytes) != len(strips_bytes) or len(serial_bytes) % 8:
        return float('inf')
    count = len(serial_bytes) // 8
    ref = struct.unpack('<' + 'd' * count, serial_bytes)
    actual = struct.unpack('<' + 'd' * count, strips_bytes)
    return max(abs(a - b) for a, b in zip(ref, actual))


def difference_summary(serial_bytes, strips_bytes):
    if len(serial_bytes) != len(strips_bytes):
        return f'file lengths {len(serial_bytes)} vs {len(strips_bytes)}'
    first = next((i for i, (a, b) in enumerate(zip(serial_bytes, strips_bytes)) if a != b), None)
    if first is None:
        return 'identical'
    if len(serial_bytes) % 8 == 0:
        count = len(serial_bytes) // 8
        ref = struct.unpack('<' + 'd' * count, serial_bytes)
        actual = struct.unpack('<' + 'd' * count, strips_bytes)
        max_abs = max_absolute_error(serial_bytes, strips_bytes)
        return f'first differing byte={first}, max absolute difference={max_abs:.17g}'
    return f'first differing byte={first}'


def cases(quick):
    # N=8, P=3 has 7 interior rows: uneven (3/2/2); N=9, P=4 is even (2 each).
    # The N=8 one-step case is the professor's recommended boundary/halo diagnostic.
    tests = [
        ('2', 8, '1', ['--steps', '1']),
        ('2', 8, '9', ['--steps', '9']),
        ('V', 16, '64', ['--T', '0.05']),
    ]
    if not quick:
        tests += [
            ('1', 9, '7', ['--steps', '7']),
            ('V', 8, '12', ['--steps', '12']),
            ('1', 32, '23', ['--steps', '23']),
            ('V', 16, 'variable_T', ['--T', '0.053']),
        ]
    return tests


def main():
    a = parser().parse_args()
    if a.hosts and a.hostfile:
        parser().error('choose either --hosts or --hostfile')
    exe = Path(a.exe).resolve()
    if not exe.is_file():
        parser().error(f'compile before testing: {exe}')
    if not shutil.which(a.mpi):
        parser().error(f'MPI launcher not found: {a.mpi}')
    ranks = [int(n) for n in a.ranks.split(',')]
    if not ranks or any(n < 1 for n in ranks) or len(set(ranks)) != len(ranks):
        parser().error('--ranks must list distinct positive integers')

    report = Path(a.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ('case', 'N', 'steps_requested', 'P', 'r', 'status',
                  'field_bytes', 'serial_sha256', 'strips_sha256',
                  't_total_s', 't_comp_s', 't_comm_s', 'max_abs_diff')
    # Write each comparison immediately to support interrupted experiment runs.
    already_written = report.is_file() and report.stat().st_size > 0
    with report.open('a', newline='') as file, tempfile.TemporaryDirectory(prefix='heat_strips_test_') as folder:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not already_written:
            writer.writeheader()
        tmp = Path(folder)
        number = 0
        mismatches = 0
        largest_difference = 0.0
        for kind, n, label, time_args in cases(a.quick):
            serial = tmp / f'serial_{kind}_{n}_{label}.bin'
            args = ['--case', kind, '--N', str(n), '--r', '0.2',
                    *time_args]
            base = cmd_output([str(exe), '--decomp', 'serial', *args, '--out', str(serial)], a.timeout)
            reference = serial.read_bytes()
            expected = 8 * (n+1)**2
            if len(reference) != expected:
                raise AssertionError(f'serial output length is wrong for N={n}')
            for P in ranks:
                if P > n-1:
                    print(f'SKIP P={P} N={n}: more ranks than interior rows', file=sys.stderr)
                    continue
                mpi = [a.mpi]
                if a.hostfile:
                    mpi += ['--hostfile', a.hostfile, '--map-by', 'node']
                if a.hosts:
                    mpi += ['--host', a.hosts, '--map-by', 'node']
                mpi += ['-np', str(P)]
                if a.tailscale:
                    mpi += ['--mca', 'pml', 'ob1',
                            '--mca', 'oob_tcp_if_include', 'tailscale0',
                            '--mca', 'btl_tcp_if_include', 'tailscale0',
                            '--mca', 'btl', 'self,tcp']
                strips = tmp / f'strips_{kind}_{n}_{label}_p{P}.bin'
                parallel = cmd_output([*mpi, str(exe), '--decomp', 'strips',
                                       *args, '--out', str(strips)], a.timeout)
                actual = strips.read_bytes()
                passed = reference == actual
                delta = 0.0 if passed else max_absolute_error(reference, actual)
                if not passed:
                    mismatches += 1
                    largest_difference = max(largest_difference, delta)
                record = dict(case=kind, N=n, steps_requested=label, P=P, r='0.2',
                              status='PASS' if passed else 'FAIL', field_bytes=len(actual),
                              serial_sha256=sha256(reference), strips_sha256=sha256(actual),
                              t_total_s=parallel.get('t_total', ''),
                              t_comp_s=parallel.get('t_comp', ''),
                              t_comm_s=parallel.get('t_comm', ''), max_abs_diff=f'{delta:.17g}')
                writer.writerow(record)
                file.flush()
                number += 1
                if not passed and not a.mixed_arch_report:
                    raise AssertionError(f'FAIL case={kind} N={n} P={P}: '
                                         f'{difference_summary(reference, actual)}\n'
                                         f'serial={base}\nstrips={parallel}')
                if not passed:
                    print(f'MIXED-ARCH DIFFERENCE case={kind} N={n} P={P}: '
                          f'{difference_summary(reference, actual)}')
                if parallel.get('P') != str(P) or parallel.get('decomp') != 'strips':
                    raise AssertionError(f'Incorrect MPI summary: {parallel}')
                if passed and kind == 'V':
                    for error_norm in ('err_inf', 'err_l2'):
                        if base[error_norm] != parallel[error_norm]:
                            raise AssertionError(f'Mismatching {error_norm} despite identical fields')
                if passed:
                    print(f'PASS case={kind} N={n} steps={parallel["steps"]} P={P} bytes={expected}')
    if mismatches:
        print(f'MIXED-ARCH REPORT: {mismatches} mismatches among {number} cases; '
              f'largest absolute difference={largest_difference:.17g}; report={report}')
    else:
        print(f'PASS: {number} byte-for-byte strip comparisons, report={report}')


if __name__ == '__main__':
    main()
