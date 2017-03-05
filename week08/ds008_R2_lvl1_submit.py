#!/usr/bin/env python

from os import system
from os.path import abspath, join

subjs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
         'sub-06', 'sub-07', 'sub-09', 'sub-10',
         'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15']

for vec_type in ['RTamp', 'RTdur']:
    salo_dir = '/scratch/PSB6351_2017/students/salo/'
    work_dir = join(salo_dir, 'working/week08/ds008_R2_frstlvl_{0}/'.format(vec_type))
    out_dir = join(salo_dir, 'data/frstlvl_{0}/'.format(vec_type))
    script = join(salo_dir, 'week08/ds008_R2_lvl1_{0}_HW.py'.format(vec_type))
    crash_dir = join(salo_dir, 'crash/week08/frstlvl/')
    for i, sid in enumerate(subjs):
        err_file = join(crash_dir, vec_type, 'ds008_R2_lvl1_err_{0}'.format(sid))
        out_file = join(crash_dir, vec_type, 'ds008_R2_lvl1_out_{0}'.format(sid))
        convertcmd = ' '.join(['python', script, '-s', sid, '-o', out_dir, '-w', work_dir])
        script_file = abspath(join(vec_type, 'ds008_R2_lvl1-{1}.sh'.format(sid)))
        with open(script_file, 'wt') as fp:
            fp.writelines(['#!/bin/bash\n', convertcmd])
        outcmd = 'bsub -J atm-ds008lvl1-{0} -q PQ_nbc -e {1} -o {2} < {3}' % (sid, err_file, out_file, script_file)
        system(outcmd)
        continue
