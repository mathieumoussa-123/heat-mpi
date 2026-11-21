#!/usr/bin/env python3
"""Non-MPI unit/integration tests of Part 4 scripts. No benchmark results invented."""
from __future__ import annotations

import csv
import importlib.util
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PROJECT = SCRIPTS.parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Part4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = module('run_perf')
        cls.model = module('model_perf')
        cls.analyzer = module('analyze_perf')

    def test_partition_and_placement(self):
        entries = [('mathieu',4),('naji',4),('chris',4)]
        self.assertEqual(self.model.inventory('heat-mathieu:4,heat-naji:4,heat-chris:4','node',8),
                         ['mathieu','naji','chris','mathieu','naji','chris','mathieu','naji'])
        self.assertEqual(self.model.inventory('heat-mathieu:4,heat-naji:4,heat-chris:4','slot',8),
                         ['mathieu']*4+['naji']*4)
        self.assertEqual(self.model.grid_dims(8), (4,2))
        self.assertEqual([self.model.partition(8,3,p) for p in range(3)], [3,2,2])
        self.assertRaises(ValueError, self.runner.host_entries, 'heat-mathieu:5')

    def test_heat_output_parser(self):
        self.assertEqual(self.runner.parse_heat_output('decomp=serial t_total=2.5 t_comp=2.5 t_comm=0')['t_total'], 2.5)
        self.assertRaises(ValueError,self.runner.parse_heat_output,'garbled text')

    def test_append_resume_and_corrupt_data(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'raw.csv'
            row={k:'' for k in self.runner.FIELDS}
            row.update(experiment='baseline',vm='mathieu',N=512,P=1,decomp='serial',case='2',placement='local',
                       repetition=1,steps=200,r='0.2',transport='none',hosts='',
                       t_total_s='2.0',t_comp_s='2.0',t_comm_s='0')
            self.runner.append_row(path,row)
            rows=self.runner.load_records(path)
            self.assertEqual(len(rows),1)
            self.runner.append_row(path,row)
            self.assertRaises(ValueError,self.runner.load_records,path)

    def test_model_one_and_two_ranks(self):
        costs={vm:[(512,1e-8),(8192,1e-8)] for vm in ('mathieu','naji','chris')}
        links={frozenset(('mathieu','naji')):(.06,1e-5),
               frozenset(('mathieu','chris')):(.05,1e-5),
               frozenset(('naji','chris')):(.04,1e-5)}
        h='heat-mathieu:4,heat-naji:4,heat-chris:4'
        one,_,comm=self.model.estimate(512,1,'strips','node',h,costs,links)
        self.assertAlmostEqual(comm,0)
        self.assertEqual(one['remote_count'],0)
        two,_,comm=self.model.estimate(512,2,'strips','node',h,costs,links)
        self.assertEqual(two['remote_count'],1)
        self.assertGreater(comm,0)
        scenario,_,_=self.model.estimate(512,2,'strips','node',h,costs,links,scenario_ts=1e-6)
        self.assertLess(scenario['total'],two['total'])

    def test_import_part3_and_skip_previous(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'part3.csv'
            fields=['P','N','decomp','repetition','steps','r','placement','t_total_s','t_comp_s','t_comm_s']
            with source.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields)
                w.writeheader()
                for P in (4,8):
                    for N in (512,2048,8192):
                        for dec in ('strips','blocks'):
                            for rep in (1,2,3):
                                w.writerow(dict(P=P,N=N,decomp=dec,repetition=rep,steps=200,r=0.2,
                                                placement='node',t_total_s=2.1,t_comp_s=.1,t_comm_s=2))
            from argparse import Namespace
            opts=Namespace(csv=str(source),out_root=td,
                           hosts='heat-mathieu:4,heat-naji:4,heat-chris:4')
            self.runner.import_part3(opts)
            self.runner.import_part3(opts)
            data=self.runner.load_records(Path(td)/'strong'/'raw.csv')
            self.assertEqual(len(data),36)

    def test_fake_serial_run_and_mpi_resume(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td)
            heat=td/'fake_heat'
            heat.write_text('#!/usr/bin/env python3\nprint("decomp=strips P=4 N=512 steps=200 t_total=2.0 t_comp=0.5 t_comm=1.5")\n')
            heat.chmod(0o755)
            mpi=td/'fake_mpirun'
            mpi.write_text('#!/usr/bin/env python3\nimport os,subprocess,sys\na=sys.argv[1:]\nfor i,x in enumerate(a):\n if x.endswith("/fake_heat"):\n  sys.exit(subprocess.call(a[i:]))\nsys.exit(2)\n')
            mpi.chmod(0o755)
            root=td/'results'
            def call(*args):
                return subprocess.run([sys.executable,str(SCRIPTS/'run_perf.py'),*args],
                                      text=True,capture_output=True)
            cmd=('strong','--hosts','heat-mathieu:4,heat-naji:4,heat-chris:4',
                 '--Ns','512','--Ps','4','--decomps','strips','--steps','200',
                 '--repeats','3','--mpi',str(mpi),'--exe',str(heat),
                 '--out-root',str(root),'--execute')
            result=call(*cmd)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(self.runner.load_records(root/'strong'/'raw.csv')),3)
            result=call(*cmd)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(len(self.runner.load_records(root/'strong'/'raw.csv')),3)
            bad=call('strong','--hosts','heat-mathieu:5','--Ns','512',
                     '--Ps','4','--out-root',str(root))
            self.assertNotEqual(bad.returncode,0)

if __name__=='__main__':
    unittest.main(verbosity=2)
