#!/usr/bin/env python3
"""Record VM metadata for reproducibility; wifi/ethernet is supplied by the user.
Do not store keys, secrets, or raw Tailscale JSON (it can contain peer details).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import shutil
import socket
import subprocess
from pathlib import Path


def run(*args):
    if not shutil.which(args[0]):
        return '(not installed)'
    try:
        proc = subprocess.run(list(args), capture_output=True, text=True, timeout=10,
                              check=False)
        return (proc.stdout or proc.stderr).strip()[:12000]
    except (subprocess.SubprocessError, OSError) as e:
        return '(unavailable: ' + str(e) + ')'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vm', required=True, choices=('mathieu','naji','chris'))
    p.add_argument('--connection', required=True, choices=('wifi','ethernet','other','unknown'))
    p.add_argument('--out-root', default='results/part4')
    args = p.parse_args()
    host = socket.gethostname().split('.')[0]
    if host != 'heat-'+args.vm:
        p.error(f'wrong VM: --vm {args.vm}, actual hostname {host}')
    d = dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),
             vm=args.vm, hostname=host, network_type_user_reported=args.connection,
             python=platform.python_version(), kernel=platform.platform(),
             cpu_count_os=run('nproc'), lscpu=run('lscpu'),
             memory=run('free','-h'), mpi=run('mpirun','--version'),
             tailscale_version=run('tailscale','version'))
    path = Path(args.out_root)/'metadata'/f'{args.vm}.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(f'SAVED {path}')
    print('Record direct-vs-DERP peer connection using tailscale ping before/during runs; '
          'this metadata snapshot does not establish it.')

if __name__ == '__main__':
    main()
