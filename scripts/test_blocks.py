#!/usr/bin/env python3
"""Part 3: verify MPI Cartesian blocks against the Part 1 serial reference.

Run with --hosts localhost:4 for local P<=4 correctness; multi-VM --tailscale.
This script tests byte-for-byte equality, not approximate floating-point equality.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', default='build/heat')
    parser.add_argument('--mpi', default='mpirun')
    parser.add_argument('--hosts', help='e.g. localhost:4 or heat-mathieu:4,heat-naji:4,heat-chris:4')
    parser.add_argument('--hostfile')
    parser.add_argument('--tailscale', action='store_true')
    parser.add_argument('--ranks', default='1,2,3,4', help='e.g. 1,2,3,4,6,8')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--also-strips', action='store_true',
                        help='Additionally compare MPI strips with serial and blocks')
    parser.add_argument('--mixed-arch-report', action='store_true',
                        help='Continue after non-bitwise-equal output and record max absolute difference')
    parser.add_argument('--timeout', type=int, default=600, help='seconds per MPI command')
    parser.add_argument('--report', default='results/part3/blocks_correctness.csv')
    return parser


def run(command, timeout):
    try:
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f'TIMEOUT after {timeout}s: {" ".join(command)}') from error
    if completed.returncode:
        raise RuntimeError(f'FAILED ({completed.returncode}): {" ".join(command)}\n'
                           f'STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}')
    summaries = [line.strip() for line in completed.stdout.splitlines()
                 if line.startswith('decomp=')]
    if len(summaries) != 1:
        raise RuntimeError(f'expected exactly one rank-zero summary, got {summaries!r}\n'
                           f'STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}')
    return dict(piece.split('=', 1) for piece in summaries[0].split())


def test_matrix(quick):
    suite = [
        ('2', 8, 'steps1', ['--steps', '1']),
        ('2', 8, 'steps9', ['--steps', '9']),
        ('V', 16, 'T005', ['--T', '0.05']),
    ]
    if not quick:
        suite += [
            ('1', 9, 'steps7', ['--steps', '7']),
            ('V', 8, 'steps12', ['--steps', '12']),
            ('1', 32, 'steps23', ['--steps', '23']),
            ('V', 16, 'T0053', ['--T', '0.053']),
            ('2', 9, 'steps17', ['--steps', '17']),
        ]
    return suite


def shape(P):
    # MPI_Dims_create(P, 2, [0, 0]) normally returns the most balanced factors.
    # This is ONLY used to skip empty-domain tests, not to configure actual MPI.
    minor = 1
    for factor in range(1, int(P ** 0.5) + 1):
        if P % factor == 0:
            minor = factor
    return P // minor, minor


def mpi_prefix(args, P):
    command = [args.mpi]
    if args.hostfile:
        command += ['--hostfile', args.hostfile, '--map-by', 'node']
    elif args.hosts:
        command += ['--host', args.hosts, '--map-by', 'node']
    command += ['-np', str(P)]
    if args.tailscale:
        command += ['--mca', 'pml', 'ob1',
                    '--mca', 'oob_tcp_if_include', 'tailscale0',
                    '--mca', 'btl_tcp_if_include', 'tailscale0',
                    '--mca', 'btl', 'self,tcp']
    return command


def first_difference(a, b):
    if len(a) != len(b):
        return f'length mismatch {len(a)} vs {len(b)}'
    location = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if location is None:
        return 'identical'
    if len(a) % 8:
        return f'first differing byte {location}'
    ref = struct.iter_unpack('<d', a)
    actual = struct.iter_unpack('<d', b)
    deviation = max(abs(x[0] - y[0]) for x, y in zip(ref, actual))
    return f'first differing byte {location}; max |error|={deviation:.17g}'


def max_absolute_difference(a, b):
    if len(a) != len(b) or len(a) % 8:
        return float('inf')
    return max(abs(x[0] - y[0]) for x, y in zip(struct.iter_unpack('<d', a), struct.iter_unpack('<d', b)))


def main():
    parser = options()
    a = parser.parse_args()
    if a.hosts and a.hostfile:
        parser.error('use either --hosts or --hostfile')
    if a.timeout < 1:
        parser.error('--timeout must be positive')
    exe = Path(a.exe).resolve()
    if not exe.is_file():
        parser.error(f'heat executable does not exist: {exe}')
    if not shutil.which(a.mpi):
        parser.error(f'MPI launcher not found: {a.mpi}')
    try:
        rank_counts = [int(x) for x in a.ranks.split(',')]
        if not rank_counts or any(p < 1 for p in rank_counts) or len(set(rank_counts)) != len(rank_counts):
            raise ValueError
    except ValueError:
        parser.error('--ranks must be a comma-separated list of distinct positive integers')
    report = Path(a.report).resolve()
    report.parent.mkdir(parents=True, exist_ok=True)
    # No accidental duplicate records in a previously completed evidence file.
    if report.exists() and report.stat().st_size:
        parser.error(f'{report} already contains data; choose a new --report filename')

    columns = ['case', 'N', 'time_mode', 'P', 'blocks_shape', 'status', 'bytes',
               'serial_sha256', 'blocks_sha256', 'strips_sha256', 't_total_s',
               't_comp_s', 't_comm_s', 'err_inf', 'err_l2', 'max_abs_diff']
    passed = 0
    failed = 0
    largest_difference = 0.0
    with tempfile.TemporaryDirectory(prefix='heat_blocks_test_') as folder, report.open('w', newline='') as output:
        temp = Path(folder)
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        output.flush()
        for case, N, time_mode, time_args in test_matrix(a.quick):
            arguments = ['--case', case, '--N', str(N), '--r', '0.2', *time_args]
            serial_path = temp / f'serial_{case}_{N}_{time_mode}.bin'
            serial_meta = run([str(exe), '--decomp', 'serial', *arguments,
                               '--out', str(serial_path)], a.timeout)
            reference = serial_path.read_bytes()
            expected = 8 * (N + 1) ** 2
            if len(reference) != expected:
                raise AssertionError(f'wrong reference length for N={N}: {len(reference)}')
            for P in rank_counts:
                y, x = shape(P)
                if max(x, y) > N - 1:
                    print(f'SKIP N={N}, P={P}: empty Cartesian block would be needed')
                    continue
                file_path = temp / f'blocks_{case}_{N}_{time_mode}_P{P}.bin'
                prefix = mpi_prefix(a, P)
                actual_meta = run([*prefix, str(exe), '--decomp', 'blocks',
                                   *arguments, '--out', str(file_path)], a.timeout)
                actual = file_path.read_bytes()
                if actual_meta.get('P') != str(P) or actual_meta.get('decomp') != 'blocks':
                    raise AssertionError(f'unexpected blocks summary: {actual_meta}')
                matches_serial = actual == reference
                error = 0.0 if matches_serial else max_absolute_difference(reference, actual)
                if not matches_serial:
                    failed += 1
                    largest_difference = max(largest_difference, error)
                for norm in ('err_inf', 'err_l2'):
                    if matches_serial and case == 'V' and actual_meta.get(norm) != serial_meta.get(norm):
                        raise AssertionError(f'{norm} not identical for case={case} N={N} P={P}')

                strips_sha = ''
                if a.also_strips and P <= N - 1:
                    strips_path = temp / f'strips_{case}_{N}_{time_mode}_P{P}.bin'
                    strips_meta = run([*prefix, str(exe), '--decomp', 'strips',
                                       *arguments, '--out', str(strips_path)], a.timeout)
                    strips_bytes = strips_path.read_bytes()
                    if strips_bytes != reference or strips_meta.get('P') != str(P):
                        raise AssertionError('STRIPS MISMATCH ' + first_difference(reference, strips_bytes))
                    strips_sha = hashlib.sha256(strips_bytes).hexdigest()
                record = {
                    'case': case, 'N': N, 'time_mode': time_mode, 'P': P,
                    'blocks_shape': f'{y}x{x}', 'status': 'PASS' if matches_serial else 'FAIL',
                    'bytes': len(actual),
                    'serial_sha256': hashlib.sha256(reference).hexdigest(),
                    'blocks_sha256': hashlib.sha256(actual).hexdigest(),
                    'strips_sha256': strips_sha,
                    't_total_s': actual_meta.get('t_total', ''),
                    't_comp_s': actual_meta.get('t_comp', ''),
                    't_comm_s': actual_meta.get('t_comm', ''),
                    'err_inf': actual_meta.get('err_inf', ''),
                    'err_l2': actual_meta.get('err_l2', ''),
                    'max_abs_diff': f'{error:.17g}',
                }
                writer.writerow(record)
                output.flush()
                if not matches_serial:
                    print(f'MISMATCH case={case} N={N} mode={time_mode} P={P}: '
                          + first_difference(reference, actual), flush=True)
                    if not a.mixed_arch_report:
                        raise AssertionError('Block output is not byte-for-byte identical to serial')
                else:
                    passed += 1
                    print(f'PASS case={case} N={N} mode={time_mode} P={P} '
                          f'cart={y}x{x} bytes={expected}', flush=True)
    if failed:
        print(f'MIXED-ARCH REPORT: {failed} mismatches, {passed} exact comparisons; '
              f'largest absolute difference={largest_difference:.17g}; report={report}')
    else:
        print(f'PASS: {passed} byte-for-byte blocks comparisons, report={report}')


if __name__ == '__main__':
    main()
