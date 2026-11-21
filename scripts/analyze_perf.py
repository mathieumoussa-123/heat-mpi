#!/usr/bin/env python3
"""Part 4: analyze raw serial/strong/weak timings and create figures.

Can run on partial data while collecting measurements; report coverage explicitly.
Never treats missing repetitions as finished experiments.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    for item in rows:
        for field in ('N', 'P', 'steps', 'repetition'):
            item[field] = int(item[field])
        for field in ('r', 't_total_s', 't_comp_s', 't_comm_s'):
            item[field] = float(item[field])
    return rows


def save(path: Path, rows: list[dict], fields: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'WROTE {path} ({len(rows)} rows)')


def aggregate(rows: list[dict], group_fields: tuple, repeats: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in group_fields)].append(row)
    result = []
    for identity, group in sorted(grouped.items(), key=lambda x: tuple(str(v) for v in x[0])):
        # Must not silently accept duplicate repetition identities.
        ids = [row['repetition'] for row in group]
        if len(ids) != len(set(ids)):
            raise ValueError(f'duplicate repetition: {group_fields}, {identity}')
        if any((row['steps'], row['r'], row['transport'], row['hosts'], row['case']) !=
               (group[0]['steps'], group[0]['r'], group[0]['transport'], group[0]['hosts'], group[0]['case'])
               for row in group):
            raise ValueError(f'heterogeneous settings in group {identity}')
        totals = [x['t_total_s'] for x in group]
        comps = [x['t_comp_s'] for x in group]
        comms = [x['t_comm_s'] for x in group]
        med_total = statistics.median(totals)
        med_comm = statistics.median(comms)
        entry = dict(zip(group_fields, identity))
        entry.update(case=group[0]['case'], steps=group[0]['steps'], r=group[0]['r'],
                     transport=group[0]['transport'], hosts=group[0]['hosts'],
                     repeats=len(group), expected_repeats=repeats,
                     complete=len(group) >= repeats,
                     med_total_s=med_total, min_total_s=min(totals), max_total_s=max(totals),
                     spread_total_s=max(totals) - min(totals),
                     med_step_s=med_total / group[0]['steps'],
                     med_comp_s=statistics.median(comps), med_comm_s=med_comm,
                     comm_share=med_comm / med_total if med_total > 0 else float('nan'))
        result.append(entry)
    return result


def prep_charts():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def render(fig_dir: Path, baseline: list[dict], strong: list[dict], weak: list[dict]) -> None:
    plt = prep_charts()
    fig_dir.mkdir(parents=True, exist_ok=True)

    def finish(name, xlabel, ylabel, logx=False, logy=False, ideal=False):
        if ideal:
            plt.plot([1, 12], [1, 12], '--', color='gray', label='Ideal S(P)=P')
        if logx:
            plt.xscale('log', base=2)
        if logy:
            plt.yscale('log')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / name, dpi=180)
        plt.close()

    if baseline:
        for field, fname, ylabel in [
            ('med_step_s', 'serial_time_per_step.png', 'Serial time per step (s)'),
            ('seconds_per_node_step', 'serial_seconds_per_node.png', 'Seconds per interior-node update')]:
            plt.figure(figsize=(8, 5))
            for vm in sorted({row['vm'] for row in baseline}):
                seq = sorted([row for row in baseline if row['vm'] == vm], key=lambda a: a['N'])
                plt.plot([x['N'] for x in seq], [x[field] for x in seq], marker='o', label=vm)
            finish(fname, 'N (grid intervals)', ylabel, logx=True, logy=True)
    if strong:
        for N in sorted({row['N'] for row in strong}):
            these = [x for x in strong if x['N'] == N]
            for field, prefix, ylabel, ideal in [
                ('med_total_s', 'strong_runtime', '200-step total time (s)', False),
                ('speedup', 'strong_speedup', 'Speedup versus Mathieu serial', True),
                ('efficiency', 'strong_efficiency', 'Parallel efficiency', False),
                ('comm_share', 'communication_share', 'Ratio max(comm) / max(total)', False)]:
                plt.figure(figsize=(9, 5))
                plotted = False
                for placement in ('node', 'slot'):
                    for decomp in ('strips', 'blocks'):
                        seq = sorted([x for x in these if x['placement'] == placement and x['decomp'] == decomp
                                      and math.isfinite(float(x.get(field, float('nan'))))], key=lambda x:x['P'])
                        if seq:
                            plt.plot([x['P'] for x in seq], [x[field] for x in seq],
                                     marker='o', label=f'{decomp} / {placement}')
                            plotted = True
                if plotted:
                    finish(f'{prefix}_N{N}.png', 'MPI ranks P', ylabel, ideal=ideal)
                else:
                    plt.close()
    if weak:
        for field, fname, ylabel in [
            ('med_step_s', 'weak_scaling_time_per_step.png', 'Time per step (s)'),
            ('weak_efficiency', 'weak_scaling_efficiency.png', 'Weak scaling efficiency T(1)/T(P)')]:
            plt.figure(figsize=(9, 5))
            plotted = False
            for placement in ('node', 'slot'):
                for decomp in ('strips', 'blocks'):
                    seq = sorted([x for x in weak if x['placement'] == placement and x['decomp'] == decomp
                                  and math.isfinite(float(x.get(field, float('nan'))))], key=lambda x:x['P'])
                    if seq:
                        plt.plot([x['P'] for x in seq], [x[field] for x in seq],
                                 marker='o', label=f'{decomp} / {placement}')
                        plotted = True
            if plotted:
                finish(fname, 'MPI ranks P', ylabel)
            else:
                plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', default='results/part4')
    p.add_argument('--reference-vm', default='mathieu', choices=('mathieu', 'naji', 'chris'))
    p.add_argument('--expected-repeats', type=int, default=3)
    args = p.parse_args()
    root = Path(args.out_root)
    baseline_rows = []
    for vm in ('mathieu', 'naji', 'chris'):
        baseline_rows += load(root / 'baseline' / vm / 'raw.csv')
    strong_rows = load(root / 'strong' / 'raw.csv')
    weak_rows = load(root / 'weak' / 'raw.csv')
    for row in baseline_rows + strong_rows + weak_rows:
        if row['steps'] != 200 or row['r'] != 0.2 or row['case'] != '2':
            raise ValueError('Part 4 dataset should use 200 steps and r=0.2 throughout')
    baseline = aggregate(baseline_rows, ('vm', 'N'), args.expected_repeats)
    strong = aggregate(strong_rows, ('N', 'P', 'decomp', 'placement'), args.expected_repeats)
    weak = aggregate(weak_rows, ('N', 'P', 'decomp', 'placement'), args.expected_repeats)
    for row in baseline:
        row['seconds_per_node_step'] = row['med_total_s'] / (row['steps'] * (row['N'] - 1)**2)
        row['node_updates_per_s'] = 1 / row['seconds_per_node_step']
    serial = {row['N']: row['med_total_s'] for row in baseline if row['vm'] == args.reference_vm and row['complete']}
    for row in strong:
        t1 = serial.get(row['N'])
        row['reference_serial_s'] = t1 if t1 is not None else ''
        row['speedup'] = t1 / row['med_total_s'] if t1 is not None and row['med_total_s'] else float('nan')
        row['efficiency'] = row['speedup'] / row['P']
    weak_t1 = {(row['decomp'], row['placement']): row['med_step_s'] for row in weak
               if row['P'] == 1 and row['complete']}
    for row in weak:
        row['work_per_rank'] = ((row['N'] - 1)**2) / row['P']
        ref = weak_t1.get((row['decomp'], row['placement']))
        row['weak_efficiency'] = ref / row['med_step_s'] if ref is not None and row['med_step_s'] else float('nan')

    baseline_fields = list(baseline[0]) if baseline else ['vm', 'N', 'med_step_s']
    strong_fields = list(strong[0]) if strong else ['N', 'P', 'decomp', 'placement', 'med_total_s']
    weak_fields = list(weak[0]) if weak else ['N', 'P', 'decomp', 'placement', 'med_total_s']
    save(root / 'analysis' / 'baseline_summary.csv', baseline, baseline_fields)
    save(root / 'analysis' / 'strong_summary.csv', strong, strong_fields)
    save(root / 'analysis' / 'weak_summary.csv', weak, weak_fields)
    render(root / 'figures', [x for x in baseline if x['complete']],
           [x for x in strong if x['complete']], [x for x in weak if x['complete']])

    # Presence AND repetitions are checked; don't claim complete based on generated files alone.
    base_keys = {(x['vm'], x['N']) for x in baseline if x['complete']}
    strong_keys = {(x['N'], x['P'], x['decomp'], x['placement']) for x in strong if x['complete']}
    weak_keys = {(x['P'], x['decomp'], x['placement']) for x in weak if x['complete']}
    expected_base = {(vm,N) for vm in ('mathieu','naji','chris') for N in (512,1024,2048,4096,8192)}
    expected_strong = {(N,P,dec,place) for N in (512,2048,8192) for P in range(1,13)
                       for dec in ('strips','blocks') for place in ('node','slot')}
    expected_weak = {(P,dec,'node') for P in (1,2,3,4,6,8,12) for dec in ('strips','blocks')}
    status = (f'SERIAL: {len(base_keys & expected_base)}/{len(expected_base)} configs complete\n'
              f'STRONG: {len(strong_keys & expected_strong)}/{len(expected_strong)} configs complete\n'
              f'WEAK node: {len(weak_keys & expected_weak)}/{len(expected_weak)} configs complete\n'
              'A configuration is complete only when it has at least 3 independent runs.\n'
              'Communication share = median(max-rank communication) / median(max-rank total); '
              'these maxima may arise on different ranks.\n'
              'All total-time plots are based on the timed stepping loop, excluding startup and output.\n')
    (root / 'analysis' / 'coverage.txt').write_text(status)
    print(status)

if __name__ == '__main__':
    main()
