#!/usr/bin/env python3
"""Part 4 reproducible MPI performance runner (prediction-only unless --execute).

Baseline: run locally on EACH VM; strong/weak: launch from Mathieu.
Raw results are append-only, validated and resume by experiment identity.
The default TCP transport exactly matches the validated Part 3 benchmark.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

FIELDS = ['experiment', 'vm', 'N', 'P', 'decomp', 'case', 'placement', 'repetition',
          'steps', 'r', 'transport', 'hosts', 't_total_s', 't_comp_s', 't_comm_s',
          'wall_s', 'source', 'utc']
KINDS = ('baseline', 'strong', 'weak')
NAMES = {'mathieu': 'heat-mathieu', 'naji': 'heat-naji', 'chris': 'heat-chris'}


def integers(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(',') if x.strip()]
    if not values or any(x <= 0 for x in values) or len(values) != len(set(values)):
        raise ValueError(f'invalid distinct positive integers: {text}')
    return values


def host_entries(hosts: str) -> list[tuple[str, int]]:
    if not hosts:
        raise ValueError('distributed runs require --hosts')
    entries = []
    for item in hosts.split(','):
        match = re.fullmatch(r'([A-Za-z0-9][\w.-]*):(\d+)', item.strip())
        if not match or int(match.group(2)) < 1:
            raise ValueError(f'hosts must be hostname:slots,...; invalid {item!r}')
        entries.append((match.group(1), int(match.group(2))))
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError('duplicate host name')
    # These three VMs have four vCPUs each; enforce professor's physical limit.
    if any(slots > 4 for _, slots in entries):
        raise ValueError('each team VM has 4 vCPUs; do not allocate more than four slots')
    return entries


def key(row: dict) -> tuple:
    return tuple(str(row[k]) for k in
                 ('experiment', 'vm', 'N', 'P', 'decomp', 'case', 'placement',
                  'repetition', 'steps', 'r', 'transport', 'hosts'))


def load_records(path: Path) -> dict[tuple, dict]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    with path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f'invalid CSV header in {path}: {reader.fieldnames}')
        rows = {}
        for number, row in enumerate(reader, 2):
            try:
                for f in ('N', 'P', 'repetition', 'steps'):
                    if int(row[f]) < 1:
                        raise ValueError(f'{f} must be positive')
                if not 0 < float(row['r']) <= 0.25:
                    raise ValueError('bad r')
                for f in ('t_total_s', 't_comp_s', 't_comm_s'):
                    if not (math.isfinite(float(row[f])) and float(row[f]) >= 0):
                        raise ValueError('invalid timing')
                identity = key(row)
                if identity in rows:
                    raise ValueError('duplicate experiment key')
                rows[identity] = row
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
    return rows


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    empty = not path.exists() or path.stat().st_size == 0
    with path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if empty:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def parse_heat_output(stdout: str) -> dict[str, float]:
    lines = [line for line in stdout.splitlines()
             if 't_total=' in line and 't_comm=' in line]
    if len(lines) != 1:
        raise ValueError(f'expected exactly one heat summary line, got {len(lines)}')
    parts = dict(piece.split('=', 1) for piece in lines[0].split() if '=' in piece)
    for field in ('t_total', 't_comp', 't_comm'):
        if field not in parts:
            raise ValueError(f'heat output missing {field}')
    times = {f: float(parts[f]) for f in ('t_total', 't_comp', 't_comm')}
    if any(not math.isfinite(x) or x < 0 for x in times.values()):
        raise ValueError('non-finite/negative time')
    return times


def mpi_command(args, P: int, decomp: str, N: int, entries) -> list[str]:
    cmd = [args.mpi, '--host', args.hosts, '--map-by', args.placement, '-np', str(P)]
    if args.transport == 'tcp':
        cmd += ['--mca', 'pml', 'ob1',
                '--mca', 'oob_tcp_if_include', 'tailscale0',
                '--mca', 'btl_tcp_if_include', 'tailscale0',
                '--mca', 'btl', 'self,tcp']
    elif args.transport == 'hybrid-vader':
        cmd += ['--mca', 'pml', 'ob1',
                '--mca', 'oob_tcp_if_include', 'tailscale0',
                '--mca', 'btl_tcp_if_include', 'tailscale0',
                '--mca', 'btl', 'self,vader,tcp']
    cmd += [str(Path(args.exe).resolve()), '--decomp', decomp,
            '--case', args.case, '--N', str(N), '--r', str(args.r),
            '--steps', str(args.steps)]
    return cmd


def run_jobs(args, experiment: str) -> None:
    if args.steps <= 0 or args.repeats < 3 or args.timeout <= 0 or not 0 < args.r <= 0.25:
        raise ValueError('--steps>0, --repeats>=3, --timeout>0, and 0<r<=0.25 required')
    entries = []
    if experiment != 'baseline':
        entries = host_entries(args.hosts)
    else:
        actual = socket.gethostname().split('.')[0].lower()
        expected = NAMES[args.vm]
        if args.execute and actual != expected and not args.allow_hostname_mismatch:
            raise ValueError(f'baseline labeled {args.vm} but running on {actual}; expected {expected}')

    if experiment == 'baseline':
        grid_sizes = integers(args.Ns)
        jobs = [(N, 1, 'serial', 'local') for N in grid_sizes]
        output = Path(args.out_root) / 'baseline' / args.vm / 'raw.csv'
    elif experiment == 'strong':
        sizes = integers(args.Ns)
        processes = integers(args.Ps)
        jobs = [(N, P, dec, args.placement) for N in sizes for P in processes
                for dec in args.decomps.split(',')]
        output = Path(args.out_root) / 'strong' / 'raw.csv'
    else:
        processes = integers(args.Ps)
        jobs = [(round(1 + args.base_side * math.sqrt(P)), P, dec, args.placement)
                for P in processes for dec in args.decomps.split(',')]
        output = Path(args.out_root) / 'weak' / 'raw.csv'
    if experiment == 'weak' and args.base_side <= 0:
        raise ValueError('--base-side must be positive')
    if experiment != 'baseline':
        if any(dec not in ('strips', 'blocks') for _, _, dec, _ in jobs):
            raise ValueError('supported decompositions are strips,blocks')
        if max(P for _, P, _, _ in jobs) > sum(slots for _, slots in entries):
            raise ValueError('requested more ranks than declared slots')
        for N, P, dec, _ in jobs:
            if P > N - 1:
                raise ValueError(f'cannot divide N={N} into P={P} nonempty {dec} regions')
    existing = load_records(output)
    pending = 0
    for N, P, decomp, placement in jobs:
        for repeat in range(1, args.repeats + 1):
            row = dict(experiment=experiment, vm=args.vm if experiment == 'baseline' else '',
                       N=N, P=P, decomp=decomp, case=args.case, placement=placement, repetition=repeat,
                       steps=args.steps, r=str(args.r),
                       transport='none' if experiment == 'baseline' else args.transport,
                       hosts='' if experiment == 'baseline' else args.hosts)
            identity = key(row)
            if identity in existing:
                if args.verbose:
                    print(f'SKIP {experiment} N={N} P={P} {decomp} {placement} #{repeat}', flush=True)
                continue
            pending += 1
            if experiment == 'baseline':
                command = [str(Path(args.exe).resolve()), '--decomp', 'serial',
                           '--case', args.case, '--N', str(N), '--r', str(args.r),
                           '--steps', str(args.steps)]
            else:
                command = mpi_command(args, P, decomp, N, entries)
            print(f'{"RUN" if args.execute else "PLAN"} {experiment} N={N} P={P} '
                  f'{decomp} {placement} repetition={repeat}/{args.repeats}', flush=True)
            if not args.execute:
                if args.verbose:
                    print('  ' + shlex.join(command), flush=True)
                continue
            if not Path(args.exe).exists():
                raise ValueError(f'executable not found: {args.exe}')
            if experiment != 'baseline' and not shutil.which(args.mpi):
                raise ValueError('mpirun not found')
            import time
            begin = time.monotonic()
            log = output.parent / 'logs' / f'{experiment}_{args.vm if experiment=="baseline" else placement}_N{N}_P{P}_{decomp}_rep{repeat}.txt'
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout,
                                        check=False)
                duration = time.monotonic() - begin
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text('COMMAND: ' + shlex.join(command) + '\nRETURN CODE: ' +
                               str(result.returncode) + '\nSTDOUT:\n' + result.stdout +
                               '\nSTDERR:\n' + result.stderr, encoding='utf-8')
                if result.returncode:
                    raise RuntimeError(f'process failed rc={result.returncode}; see {log}')
                times = parse_heat_output(result.stdout)
                row.update(t_total_s=f'{times["t_total"]:.17g}',
                           t_comp_s=f'{times["t_comp"]:.17g}',
                           t_comm_s=f'{times["t_comm"]:.17g}',
                           wall_s=f'{duration:.6f}', source='part4',
                           utc=dt.datetime.now(dt.timezone.utc).isoformat())
                append_row(output, row)
                existing[identity] = row
                print(f'SAVED N={N} P={P} {decomp} total={times["t_total"]:.5g}s '
                      f'comm={times["t_comm"]:.5g}s -> {output}', flush=True)
            except subprocess.TimeoutExpired as e:
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text('TIMEOUT: ' + shlex.join(command) + '\n' + str(e), encoding='utf-8')
                raise RuntimeError(f'timeout after {args.timeout}s; see {log}') from e
    print(f'{"COMPLETE" if args.execute else "PLAN COMPLETE"}: {experiment}; '
          f'{len(existing)} saved rows in {output}; {pending} pending at start', flush=True)


def import_part3(args) -> None:
    source = Path(args.csv)
    dest = Path(args.out_root) / 'strong' / 'raw.csv'
    host_entries(args.hosts)
    existing = load_records(dest)
    new = []
    expected = {'P', 'N', 'decomp', 'repetition', 'steps', 'r', 'placement',
                't_total_s', 't_comp_s', 't_comm_s'}
    with source.open(newline='') as f:
        reader = csv.DictReader(f)
        if not expected.issubset(set(reader.fieldnames or [])):
            raise ValueError('not a valid Part 3 raw benchmark CSV')
        for line, old in enumerate(reader, 2):
            try:
                if old['placement'] != 'node' or int(old['steps']) != 200 or float(old['r']) != 0.2:
                    raise ValueError('Part 3 reuse requires steps=200, r=0.2, placement=node')
                if old['decomp'] not in ('strips', 'blocks'):
                    raise ValueError('invalid decomposition')
                if int(old['P']) not in (4, 8) or int(old['N']) not in (512, 2048, 8192):
                    raise ValueError('unexpected P or N')
                if int(old['repetition']) not in (1, 2, 3):
                    raise ValueError('unexpected repetition')
                timings = [float(old[x]) for x in ('t_total_s', 't_comp_s', 't_comm_s')]
                if not all(math.isfinite(x) and x >= 0 for x in timings):
                    raise ValueError('bad timing')
                row = dict(experiment='strong', vm='', N=int(old['N']), P=int(old['P']),
                           decomp=old['decomp'], case='2', placement='node', repetition=int(old['repetition']),
                           steps=200, r='0.2', transport='tcp', hosts=args.hosts,
                           t_total_s=old['t_total_s'], t_comp_s=old['t_comp_s'],
                           t_comm_s=old['t_comm_s'], wall_s='', source='part3-verified', utc='')
                identity = key(row)
                if identity in existing:
                    previous = existing[identity]
                    if any(abs(float(previous[k]) - float(row[k])) > 1e-9
                           for k in ('t_total_s', 't_comp_s', 't_comm_s')):
                        raise ValueError('conflicts with existing result')
                    continue
                if any(key(item) == identity for item in new):
                    raise ValueError('duplicate Part 3 row')
                new.append(row)
            except (KeyError, ValueError) as e:
                raise ValueError(f'{source}:{line}: {e}') from e
    # Import is all-or-nothing for validity and expected 36 independent runs.
    with source.open(newline='') as handle:
        original = sum(1 for _ in csv.DictReader(handle))
    if original != 36:
        raise ValueError(f'Part 3 import requires complete 36-row data, got {original}')
    expected_keys = {(P, N, dec, rep) for P in (4, 8)
                     for N in (512, 2048, 8192) for dec in ('strips', 'blocks')
                     for rep in (1, 2, 3)}
    with source.open(newline='') as handle:
        supplied_keys = {(int(row['P']), int(row['N']), row['decomp'],
                          int(row['repetition'])) for row in csv.DictReader(handle)}
    if supplied_keys != expected_keys:
        raise ValueError('Part 3 source does not contain exactly the expected 36 combinations')
    for row in new:
        append_row(dest, row)
    print(f'IMPORTED {len(new)} rows (skipped {original - len(new)} already present) -> {dest}')


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--out-root', default='results/part4')
    common.add_argument('--exe', default='build/heat')
    common.add_argument('--steps', type=int, default=200)
    common.add_argument('--repeats', type=int, default=3)
    common.add_argument('--r', type=float, default=0.2)
    common.add_argument('--case', default='2', choices=('1', '2', 'V'))
    common.add_argument('--timeout', type=int, default=9000)
    common.add_argument('--execute', action='store_true', help='actually run; otherwise plan only')
    common.add_argument('--verbose', action='store_true')
    baseline = sub.add_parser('baseline', parents=[common])
    baseline.add_argument('--vm', choices=tuple(NAMES), required=True)
    baseline.add_argument('--Ns', default='512,1024,2048,4096,8192')
    baseline.add_argument('--allow-hostname-mismatch', action='store_true',
                          help='testing only; never use to label team results')
    distributed = argparse.ArgumentParser(add_help=False)
    distributed.add_argument('--hosts', required=True)
    distributed.add_argument('--mpi', default='mpirun')
    distributed.add_argument('--transport', choices=('tcp', 'hybrid-vader'), default='tcp',
                             help='tcp matches Part 3; hybrid-vader is separate Open MPI 4 experiment')
    distributed.add_argument('--placement', choices=('node', 'slot'), default='node')
    distributed.add_argument('--decomps', default='strips,blocks')
    strong = sub.add_parser('strong', parents=[common, distributed])
    strong.add_argument('--Ns', default='512,2048,8192')
    strong.add_argument('--Ps', default=','.join(str(x) for x in range(1,13)))
    weak = sub.add_parser('weak', parents=[common, distributed])
    weak.add_argument('--Ps', default='1,2,3,4,6,8,12')
    weak.add_argument('--base-side', type=int, default=1024)
    importing = sub.add_parser('import-part3')
    importing.add_argument('--csv', required=True)
    importing.add_argument('--hosts', required=True)
    importing.add_argument('--out-root', default='results/part4')
    return p


def main():
    args = build_parser().parse_args()
    try:
        if args.action == 'import-part3':
            import_part3(args)
        else:
            run_jobs(args, args.action)
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
