#!/usr/bin/env python3
"""Part 3: predict and optionally measure MPI strip vs Cartesian block communication.

Defaults to a SAFE prediction-only run. To perform the professor's measurements,
pass --measure --hosts ... --tailscale and ensure all three VMs have sufficient memory.

The fitted t_w in Part 0 is seconds/BYTE; each heat value is eight bytes.
Predictions use the median of the three INTER-VM fits, not same-VM SHM results.
This simplified representative-link model is not a prediction of all heterogeneous
Tailscale / intra-VM links and cannot substitute for actual MPI measurements.
"""
from __future__ import annotations

import argparse
import csv
import math
import shutil
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path


def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', default='build/heat')
    p.add_argument('--mpi', default='mpirun')
    p.add_argument('--fit-summary', default='data/part0_corrected_fit_summary.csv')
    p.add_argument('--out-dir', default='results/part3')
    p.add_argument('--Ns', default='512,2048,8192', help='grid intervals to test')
    p.add_argument('--ranks', default='4,8', help='MPI process counts')
    p.add_argument('--steps', type=int, default=500, help='fixed steps for measured runs')
    p.add_argument('--r', type=float, default=0.2)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--measure', action='store_true', help='ACTUALLY launch expensive MPI runs')
    p.add_argument('--hosts', help='e.g. heat-mathieu:4,heat-naji:4,heat-chris:4')
    p.add_argument('--hostfile')
    p.add_argument('--map-by', choices=('node', 'slot'), default='node')
    p.add_argument('--tailscale', action='store_true')
    p.add_argument('--timeout', type=int, default=3600,
                   help='max seconds for each MPI measurement')
    return p


def positive_ints(value):
    result = [int(x) for x in value.split(',')]
    if not result or len(result) != len(set(result)) or any(x <= 0 for x in result):
        raise ValueError('must be a list of distinct positive integers')
    return result


def fitted_links(path):
    links = []
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if 'same VM' in row['experiment']:
                continue
            links.append((row['experiment'], float(row['t_s_us']) / 1e6,
                          float(row['t_w_s_per_byte'])))
    if len(links) != 3:
        raise ValueError(f'expected 3 inter-VM fits, got {len(links)} from {path}')
    return links


def partitions(N, groups, c):
    owned = (N - 1) // groups + int(c < ((N - 1) % groups))
    return owned


def dims(P):
    minor = max(f for f in range(1, math.isqrt(P) + 1) if P % f == 0)
    return P // minor, minor  # process rows, columns


def predicted_per_step(N, P, t_s, t_w, decomposition):
    """Critical rank's outgoing-message cost, accounting for physical edges.

    Based on Task 8.2: t_msg(bytes) = t_s + bytes*t_w.
    This adds modeled neighbor-message costs; MPI_Sendrecv and real network
    progress may overlap or serialize differently, so this is an approximation.
    """
    if decomposition == 'strips':
        if P > N - 1:
            raise ValueError('P exceeds interior rows for strips')
        return max(((int(rank > 0) + int(rank < P - 1)) *
                    (t_s + 8 * (N + 1) * t_w)) for rank in range(P))
    if decomposition != 'blocks':
        raise ValueError('unknown decomposition')
    rows, columns = dims(P)
    if max(rows, columns) > N - 1:
        raise ValueError('P creates empty Cartesian blocks')
    costs = []
    for y in range(rows):
        for x in range(columns):
            nx = partitions(N, columns, x)
            ny = partitions(N, rows, y)
            up_down = int(y > 0) + int(y < rows - 1)
            left_right = int(x > 0) + int(x < columns - 1)
            costs.append(up_down * (t_s + 8 * nx * t_w)
                         + left_right * (t_s + 8 * ny * t_w))
    return max(costs)


def summary_file(output, link_fits, Ns, ranks):
    ts = statistics.median(row[1] for row in link_fits)
    tw = statistics.median(row[2] for row in link_fits)
    out = output / 'blocks_vs_strips_predictions.csv'
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'P', 'N', 'cart_rows', 'cart_columns', 'representative_link',
            't_s_s', 't_w_s_per_byte', 't_w_s_per_double',
            'pred_strips_critical_comm_s_per_step',
            'pred_blocks_critical_comm_s_per_step',
            'task82_strip_interior_s_per_step', 'task82_block_interior_s_per_step',
            'predicted_faster'])
        writer.writeheader()
        for P in ranks:
            for N in Ns:
                rows, columns = dims(P)
                if P > N - 1 or max(rows, columns) > N - 1:
                    raise ValueError(f'P={P}, N={N}: too many processes for grid')
                strip = predicted_per_step(N, P, ts, tw, 'strips')
                blocks = predicted_per_step(N, P, ts, tw, 'blocks')
                # Task 8.2 equations: t_w measures seconds per *grid value* there;
                # measured seconds per byte must be multiplied by sizeof(double)=8.
                simplified_strip = 2 * ts + 2 * N * 8 * tw
                simplified_block = 4 * ts + 4 * (N / math.sqrt(P)) * 8 * tw
                writer.writerow(dict(P=P, N=N, cart_rows=rows, cart_columns=columns,
                                     representative_link='median of 3 inter-VM fits',
                                     t_s_s=f'{ts:.12g}', t_w_s_per_byte=f'{tw:.12g}',
                                     t_w_s_per_double=f'{8 * tw:.12g}',
                                     pred_strips_critical_comm_s_per_step=f'{strip:.12g}',
                                     pred_blocks_critical_comm_s_per_step=f'{blocks:.12g}',
                                     task82_strip_interior_s_per_step=f'{simplified_strip:.12g}',
                                     task82_block_interior_s_per_step=f'{simplified_block:.12g}',
                                     predicted_faster='blocks' if blocks < strip else 'strips'))
                print(f'P={P:<2} N={N:<5} model strips={strip:.4g}s/step '
                      f'blocks={blocks:.4g}s/step -> {"blocks" if blocks < strip else "strips"}')
    print(f'Prediction CSV: {out}')
    return out


def mpi_prefix(args, P):
    cmd = [args.mpi]
    if args.hosts:
        cmd += ['--host', args.hosts, '--map-by', args.map_by]
    elif args.hostfile:
        cmd += ['--hostfile', args.hostfile, '--map-by', args.map_by]
    cmd += ['-np', str(P)]
    if args.tailscale:
        cmd += ['--mca', 'pml', 'ob1',
                '--mca', 'oob_tcp_if_include', 'tailscale0',
                '--mca', 'btl_tcp_if_include', 'tailscale0',
                '--mca', 'btl', 'self,tcp']
    return cmd


def perform(args, output, Ns, ranks):
    if not shutil.which(args.mpi):
        raise ValueError('mpirun not found')
    if args.hosts and args.hostfile:
        raise ValueError('choose --hosts or --hostfile')
    exe = Path(args.exe).resolve()
    if not exe.exists():
        raise ValueError(f'missing heat executable {exe}')
    if args.steps < 1 or args.repeats < 1 or args.timeout < 1:
        raise ValueError('--steps, --repeats and --timeout must be positive')
    if not (0 < args.r <= 0.25):
        raise ValueError('--r must be in (0, 0.25]')
    # Keep raw files and resume after interruptions, without recomputing recorded runs.
    raw = output / 'blocks_vs_strips_raw.csv'
    fields = ['P', 'N', 'decomp', 'repetition', 'steps', 'r', 'placement',
              't_total_s', 't_comp_s', 't_comm_s']
    existing = set()
    has_existing_data = raw.exists() and raw.stat().st_size > 0
    if has_existing_data:
        with raw.open(newline='') as f:
            for entry in csv.DictReader(f):
                existing.add((int(entry['P']), int(entry['N']), entry['decomp'],
                              int(entry['repetition']), int(entry['steps']),
                              float(entry['r']), entry['placement']))
    with raw.open('a', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        if not has_existing_data:
            writer.writeheader()
        for P in ranks:
            for N in Ns:
                for decomp in ('strips', 'blocks'):
                    for repeat in range(1, args.repeats + 1):
                        key = (P, N, decomp, repeat, args.steps, args.r, args.map_by)
                        if key in existing:
                            print(f'SKIP recorded run P={P} N={N} {decomp} repeat={repeat}')
                            continue
                        cmd = [*mpi_prefix(args, P), str(exe), '--decomp', decomp,
                               '--case', '2', '--N', str(N), '--r', str(args.r),
                               '--steps', str(args.steps)]
                        print(f'RUN P={P} N={N} decomp={decomp} repeat={repeat}/{args.repeats}', flush=True)
                        try:
                            completed = subprocess.run(cmd, capture_output=True, text=True,
                                                       timeout=args.timeout, check=False)
                        except subprocess.TimeoutExpired as error:
                            raise RuntimeError(f'timed out at P={P} N={N} {decomp}: {args.timeout}s') from error
                        if completed.returncode:
                            raise RuntimeError(f'MPI failed P={P} N={N} {decomp}\n'
                                               f'STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}')
                        lines = [s for s in completed.stdout.splitlines() if s.startswith('decomp=')]
                        if len(lines) != 1:
                            raise RuntimeError(f'expected rank-zero summary, got {completed.stdout!r}')
                        record = dict(piece.split('=', 1) for piece in lines[0].split())
                        if record.get('decomp') != decomp or int(record['P']) != P or int(record['N']) != N:
                            raise RuntimeError(f'malformed summary: {record}')
                        writer.writerow(dict(P=P, N=N, decomp=decomp, repetition=repeat,
                                             steps=args.steps, r=args.r, placement=args.map_by,
                                             t_total_s=record['t_total'],
                                             t_comp_s=record['t_comp'],
                                             t_comm_s=record['t_comm']))
                        out.flush()
                        existing.add(key)
    print(f'Raw MPI timing results: {raw}')
    return raw



def summarize_measurements(output, raw, predictions, steps, requested_r, placement):
    """Report medians and min/max spreads for the professor's >=3-repeat rule."""
    if raw is None or not raw.exists():
        return None
    samples = defaultdict(list)
    with raw.open(newline='') as stream:
        for entry in csv.DictReader(stream):
            if int(entry['steps']) != steps or float(entry['r']) != requested_r or entry['placement'] != placement:
                continue
            key = (int(entry['P']), int(entry['N']), entry['decomp'])
            samples[key].append(entry)
    with predictions.open(newline='') as stream:
        predicted = {(int(e['P']), int(e['N'])): e for e in csv.DictReader(stream)}
    path = output / 'blocks_vs_strips_measured_medians.csv'
    with path.open('w', newline='') as stream:
        columns = ['P', 'N', 'decomp', 'sample_count', 'steps', 'placement',
                   'median_total_s', 'min_total_s', 'max_total_s',
                   'median_comp_s', 'median_comm_s', 'median_comm_s_per_step',
                   'model_comm_s_per_step', 'median_total_winner']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for (P, N, decomp), entries in sorted(samples.items()):
            total = [float(e['t_total_s']) for e in entries]
            comp = [float(e['t_comp_s']) for e in entries]
            comm = [float(e['t_comm_s']) for e in entries]
            other = 'strips' if decomp == 'blocks' else 'blocks'
            peer = samples.get((P, N, other), [])
            if peer:
                peer_total = statistics.median(float(e['t_total_s']) for e in peer)
                winner = decomp if statistics.median(total) < peer_total else other
            else:
                winner = 'not yet measured'
            model_key = ('pred_strips_critical_comm_s_per_step' if decomp == 'strips'
                         else 'pred_blocks_critical_comm_s_per_step')
            model_value = predicted.get((P, N), {}).get(model_key, '')
            writer.writerow(dict(P=P, N=N, decomp=decomp, sample_count=len(entries),
                                 steps=steps, placement=placement,
                                 median_total_s=f'{statistics.median(total):.12g}',
                                 min_total_s=f'{min(total):.12g}',
                                 max_total_s=f'{max(total):.12g}',
                                 median_comp_s=f'{statistics.median(comp):.12g}',
                                 median_comm_s=f'{statistics.median(comm):.12g}',
                                 median_comm_s_per_step=f'{statistics.median(comm)/steps:.12g}',
                                 model_comm_s_per_step=model_value,
                                 median_total_winner=winner))
    print(f'Measured median and spread CSV: {path}')
    return path


def create_plots(output, predictions, raw):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: CSV generated, skipping PNG plots')
        return

    with predictions.open(newline='') as f:
        pred_rows = list(csv.DictReader(f))
    observed = defaultdict(list)
    if raw is not None and raw.exists():
        with raw.open(newline='') as f:
            for row in csv.DictReader(f):
                steps = int(row['steps'])
                observed[(int(row['P']), row['decomp'], int(row['N']))].append(
                    float(row['t_comm_s']) / steps)
    for P in sorted(set(int(row['P']) for row in pred_rows)):
        subset = sorted((r for r in pred_rows if int(r['P']) == P), key=lambda x: int(x['N']))
        fig, ax = plt.subplots(figsize=(9, 5.5))
        xs = [int(row['N']) for row in subset]
        for decomp, col in [('strips', 'pred_strips_critical_comm_s_per_step'),
                             ('blocks', 'pred_blocks_critical_comm_s_per_step')]:
            ax.plot(xs, [float(r[col]) for r in subset], marker='o', linestyle='--',
                    label=f'{decomp} predicted')
            pairs = [(n, statistics.median(observed[(P, decomp, n)])) for n in xs
                     if observed[(P, decomp, n)]]
            if pairs:
                ax.plot([x for x, _ in pairs], [y for _, y in pairs], marker='s',
                        label=f'{decomp} measured median')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Grid intervals per dimension (N)')
        ax.set_ylabel('Halo exchange time per step (seconds)')
        ax.set_title(f'Strips vs blocks — communication prediction and measurements (P={P})')
        ax.grid(True, which='both', alpha=0.25)
        ax.legend()
        fig.tight_layout()
        dest = output / f'blocks_vs_strips_P{P}.png'
        fig.savefig(dest, dpi=180)
        plt.close(fig)
        print(f'Comparison plot: {dest}')


def main():
    p = cli()
    args = p.parse_args()
    try:
        Ns = positive_ints(args.Ns)
        ranks = positive_ints(args.ranks)
    except ValueError as error:
        p.error(str(error))
    fits = Path(args.fit_summary).resolve()
    if not fits.is_file():
        p.error(f'could not find corrected Part 0 fit summary: {fits}')
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print('Mode:', 'REAL MPI measurements' if args.measure else 'predictions only (no MPI launched)')
    print('Model: representative median of corrected three inter-VM fitted t_s and t_w')
    fitted = fitted_links(fits)
    for name, t_s, t_w in fitted:
        print(f'  {name}: t_s={t_s:.5g}s; t_w={t_w:.5g}s/byte')
    prediction_csv = summary_file(output, fitted, Ns, ranks)
    raw_csv = perform(args, output, Ns, ranks) if args.measure else None
    summarize_measurements(output, raw_csv, prediction_csv, args.steps, args.r, args.map_by)
    create_plots(output, prediction_csv, raw_csv)
    if not args.measure:
        print('No performance measurements were performed. Rerun with --measure after reviewing the resource cost.')


if __name__ == '__main__':
    main()
