#!/usr/bin/env python3
"""Part 4 critical-rank compute + halo cost model for strips and blocks.

Uses per-VM serial compute rates, each measured Tailscale pair's ts/tw, actual
MPI rank placement, and exact grid partitions. Same-VM TCP was NOT measured in
Part 0; its halo cost is set to zero (lower-bound assumption) unless the user
provides --local-ts-us and --local-tw-s-per-byte from a SAME-VM TCP benchmark.
Produces a second scenario replacing inter-VM latency by 1 microsecond while
holding measured bandwidth fixed. That scenario is NOT a real-cluster forecast.
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def read(path: Path):
    if not path.exists():
        return []
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def write(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f'WROTE {path}: {len(rows)} rows')


def fits(path):
    expected = ('mathieu', 'naji', 'chris')
    ans = {}
    for row in read(path):
        name = row['experiment'].lower()
        if 'same vm' in name:
            continue
        members = tuple(k for k in expected if k in name)
        if len(members) != 2:
            raise ValueError(f'cannot parse network pair {row["experiment"]!r}')
        key = frozenset(members)
        if key in ans:
            raise ValueError('duplicate link fit')
        ans[key] = float(row['t_s_us']) / 1e6, float(row['t_w_s_per_byte'])
    if len(ans) != 3:
        raise ValueError('expected exactly three measured inter-VM pair fits')
    return ans


def inventory(hosts, placement, P):
    entries = []
    for field in hosts.split(','):
        name, slots = field.rsplit(':', 1)
        low = name.lower()
        which = next((k for k in ('mathieu','naji','chris') if k in low), None)
        if which is None:
            raise ValueError(f'unknown team VM {name}')
        entries.append((which, int(slots)))
    if P > sum(slots for _, slots in entries):
        raise ValueError('insufficient slots')
    used = [0] * len(entries)
    assigned = []
    if placement == 'slot':
        for vm, slots in entries:
            assigned += [vm] * slots
        return assigned[:P]
    if placement != 'node':
        raise ValueError(f'unknown placement {placement}')
    while len(assigned) < P:
        for index, (vm, slots) in enumerate(entries):
            if len(assigned) == P:
                break
            if used[index] < slots:
                assigned.append(vm)
                used[index] += 1
    return assigned


def partition(N, groups, coord):
    count = (N - 1) // groups + int(coord < ((N - 1) % groups))
    return count


def grid_dims(P):
    col = max(x for x in range(1, math.isqrt(P) + 1) if P % x == 0)
    return P // col, col


def serial_cost_map(baselines):
    cost = defaultdict(list)
    for row in baselines:
        if row['complete'] not in ('True', 'true', True):
            continue
        cost[row['vm']].append((int(row['N']), float(row['seconds_per_node_step'])))
    return {name: sorted(values) for name, values in cost.items()}


def per_node_cost(samples, N):
    if not samples:
        raise ValueError('missing serial baseline')
    if N <= samples[0][0]:
        return samples[0][1]
    if N >= samples[-1][0]:
        return samples[-1][1]
    for (lo, cv0), (hi, cv1) in zip(samples[:-1], samples[1:]):
        if lo <= N <= hi:
            w = (math.log(N) - math.log(lo)) / (math.log(hi) - math.log(lo))
            return cv0 * (1-w) + cv1 * w
    raise AssertionError('interpolation failed')


def estimate(N, P, decomp, placement, hosts, costs, links,
             local_ts=0.0, local_tw=0.0, scenario_ts=None):
    placement_hosts = inventory(hosts, placement, P)
    rows, cols = grid_dims(P)
    per_rank = []

    def link_cost(src, dest, elements):
        if src == dest:
            return local_ts + 8 * elements * local_tw, False
        ts, tw = links[frozenset((src, dest))]
        if scenario_ts is not None:
            ts = scenario_ts
        return ts + 8 * elements * tw, True

    for rank in range(P):
        vm = placement_hosts[rank]
        if vm not in costs:
            raise ValueError(f'no serial compute baseline for {vm}')
        neighbor_messages = []
        if decomp == 'strips':
            count = partition(N, P, rank)
            owned_nodes = count * (N - 1)
            if rank > 0:
                neighbor_messages.append((rank-1, N+1))
            if rank < P-1:
                neighbor_messages.append((rank+1, N+1))
        elif decomp == 'blocks':
            cy, cx = divmod(rank, cols)
            ny = partition(N, rows, cy)
            nx = partition(N, cols, cx)
            owned_nodes = nx * ny
            if cy > 0:
                neighbor_messages.append((rank-cols, nx))
            if cy < rows-1:
                neighbor_messages.append((rank+cols, nx))
            if cx > 0:
                neighbor_messages.append((rank-1, ny))
            if cx < cols-1:
                neighbor_messages.append((rank+1, ny))
        else:
            raise ValueError(f'bad decomposition {decomp}')
        comp = per_node_cost(costs[vm], N) * owned_nodes
        comm = 0.0
        remote_count = 0
        local_count = 0
        for destination, elements in neighbor_messages:
            duration, remote = link_cost(vm, placement_hosts[destination], elements)
            comm += duration
            if remote:
                remote_count += 1
            else:
                local_count += 1
        per_rank.append(dict(rank=rank, vm=vm, comp=comp, comm=comm,
                             total=comp+comm, remote_count=remote_count,
                             local_count=local_count, owned_nodes=owned_nodes))
    critical = max(per_rank, key=lambda r:r['total'])
    return critical, max(x['comp'] for x in per_rank), max(x['comm'] for x in per_rank)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-root', default='results/part4')
    p.add_argument('--fit-summary', default='data/part0_corrected_fit_summary.csv')
    p.add_argument('--local-ts-us', type=float, default=0,
                   help='optional measured SAME-VM TCP startup latency (not SHM!)')
    p.add_argument('--local-tw-s-per-byte', type=float, default=0,
                   help='optional measured SAME-VM TCP seconds/byte (not SHM!)')
    args = p.parse_args()
    if args.local_ts_us < 0 or args.local_tw_s_per_byte < 0:
        p.error('local TCP model parameters must be nonnegative')
    root = Path(args.out_root)
    costs = serial_cost_map(read(root / 'analysis' / 'baseline_summary.csv'))
    links = fits(Path(args.fit_summary))
    sources = [('strong', read(root / 'analysis' / 'strong_summary.csv')),
               ('weak', read(root / 'analysis' / 'weak_summary.csv'))]
    fields = ['experiment','N','P','decomp','placement','steps','repeats',
              'critical_rank','critical_vm','model_total_step_s','model_comp_max_step_s',
              'model_comm_max_step_s','model_remote_messages_critical',
              'model_local_messages_critical','measured_total_step_s','measured_comm_step_s',
              'error_pct','model_1us_total_step_s','scenario_1us_delta_pct',
              'local_tcp_model']
    predictions = []
    missing = []
    for experiment, data in sources:
        for record in data:
            N, P = int(record['N']), int(record['P'])
            try:
                if not record['hosts']:
                    raise ValueError('missing exact launcher host list')
                critical, comp, comm = estimate(N, P, record['decomp'],
                                                record['placement'], record['hosts'], costs,
                                                links, args.local_ts_us/1e6,
                                                args.local_tw_s_per_byte)
                scenario, _, _ = estimate(N, P, record['decomp'],
                                          record['placement'], record['hosts'], costs,
                                          links, args.local_ts_us/1e6,
                                          args.local_tw_s_per_byte, scenario_ts=1e-6)
                measured = float(record['med_step_s'])
                predictions.append(dict(experiment=experiment, N=N,P=P,
                                        decomp=record['decomp'],placement=record['placement'],
                                        steps=record['steps'],repeats=record['repeats'],
                                        critical_rank=critical['rank'],critical_vm=critical['vm'],
                                        model_total_step_s=critical['total'],
                                        model_comp_max_step_s=comp,model_comm_max_step_s=comm,
                                        model_remote_messages_critical=critical['remote_count'],
                                        model_local_messages_critical=critical['local_count'],
                                        measured_total_step_s=measured,
                                        measured_comm_step_s=float(record['med_comm_s'])/float(record['steps']),
                                        error_pct=100*(critical['total']-measured)/measured if measured else '',
                                        model_1us_total_step_s=scenario['total'],
                                        scenario_1us_delta_pct=100*(scenario['total']/critical['total']-1)
                                        if critical['total'] else '',
                                        local_tcp_model='measured-values' if args.local_ts_us or args.local_tw_s_per_byte
                                        else 'UNMEASURED_zero_lower_bound'))
            except (ValueError, KeyError) as e:
                missing.append(f'{experiment} N={N} P={P} {record["decomp"]} '
                               f'{record["placement"]}: {e}')
    output = root / 'model'
    write(output / 'model_vs_measured.csv', predictions, fields)
    (output / 'model_missing.txt').write_text('\n'.join(missing) + ('\n' if missing else ''))
    assumptions = (
        'MODEL ASSUMPTIONS\n'
        'Task 8.2 base: T_comp ~= c*N^2/P; T_msg(bytes)=ts+bytes*tw.\n'
        'Implemented model: exact owned interior count, per-VM serial compute rate '\
        '(interpolated across N in log space), actual row/column halo sizes, '\
        'pair-specific inter-VM ts/tw, critical-rank max(comp+comm).\n'
        'Rank-to-VM placement is inferred from host list, slots and Open MPI map-by mode.\n'
        'This is an approximate critical-path model, not a cycle-accurate MPI simulator.\n'
        'Part 0 same-VM measurements are SHM, whereas Part 3/4 uses forced TCP (btl self,tcp).\n'
        'For SAME-VM TCP traffic, missing local TCP measurements cause ZERO modeled local cost: '
        'predictions are a lower-bound approximation, and model errors must be discussed.\n'
        'Optional --local-ts-us and --local-tw-s-per-byte accept independently MEASURED TCP loopback fits.\n'
        'The 1us scenario changes INTER-VM ts to 1e-6 seconds; it holds the actual '\
        'inter-VM bandwidths constant and does not model a real HPC interconnect.\n'
        'Measured solver t_total/t_comp/t_comm are independent max reductions over ranks.\n'
    )
    (output / 'model_assumptions.txt').write_text(assumptions)
    if predictions:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figs = root / 'figures'
        figs.mkdir(parents=True, exist_ok=True)
        for N in sorted({x['N'] for x in predictions if x['experiment']=='strong'}):
            series = [r for r in predictions if r['experiment']=='strong' and r['N']==N]
            plt.figure(figsize=(10,5))
            for decomp in ('strips','blocks'):
                for placement in ('node','slot'):
                    cur = sorted([r for r in series if r['decomp']==decomp and r['placement']==placement],
                                 key=lambda x:x['P'])
                    if cur:
                        plt.plot([r['P'] for r in cur],[r['measured_total_step_s'] for r in cur],
                                 marker='o', label=f'{decomp}/{placement} measured')
                        plt.plot([r['P'] for r in cur],[r['model_total_step_s'] for r in cur],
                                 linestyle='--', label=f'{decomp}/{placement} model')
            plt.xlabel('MPI ranks P')
            plt.ylabel('Time per step (s)')
            plt.title(f'Part 4 model versus measurements, N={N}')
            plt.grid(True,alpha=.25)
            plt.legend(fontsize=7,ncol=2)
            plt.tight_layout()
            plt.savefig(figs / f'model_vs_measured_N{N}.png',dpi=180)
            plt.close()
        for N in sorted({x['N'] for x in predictions if x['experiment']=='strong'}):
            plt.figure(figsize=(9,5))
            plotted=False
            for dec in ('strips','blocks'):
                cur=sorted([r for r in predictions if r['experiment']=='strong' and r['N']==N
                            and r['decomp']==dec and r['placement']=='node'],key=lambda r:r['P'])
                if cur:
                    plt.plot([r['P'] for r in cur],[r['model_total_step_s'] for r in cur],
                             marker='o',label=f'{dec}, fitted latency')
                    plt.plot([r['P'] for r in cur],[r['model_1us_total_step_s'] for r in cur],
                             linestyle='--',label=f'{dec}, ts=1 us')
                    plotted=True
            if plotted:
                plt.xlabel('MPI ranks P')
                plt.ylabel('Modeled seconds per step')
                plt.title(f'Low-latency counterfactual (bandwidth unchanged), N={N}')
                plt.grid(True,alpha=.25)
                plt.legend(fontsize=8)
                plt.tight_layout()
                plt.savefig(figs/f'latency_1us_N{N}.png',dpi=180)
            plt.close()
    print(f'MODELED {len(predictions)} configurations; {len(missing)} skipped (see model_missing.txt)')
    print(assumptions)

if __name__=='__main__':
    main()
