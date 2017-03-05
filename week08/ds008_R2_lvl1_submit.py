#!/usr/bin/env python

import os

subjs = ['sub-01']
#subjs = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
#         'sub-06', 'sub-07', 'sub-09', 'sub-10',
#         'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15']

workdir = '/scratch/PSB6351_2017/students/salo/working/week08/ds008_R2_frstlvl_RTdur/'
outdir = '/scratch/PSB6351_2017/students/salo/data/frstlvl_RTdur/'
for i, sid in enumerate(subjs):
    convertcmd = ' '.join(['python', '/scratch/PSB6351_2017/students/salo/week08/ds008_R2_lvl1_HW.py', '-s', sid, '-o', outdir, '-w', workdir])
    script_file = 'ds008_R2_lvl1-%s.sh' % sid
    with open(script_file, 'wt') as fp:
        fp.writelines(['#!/bin/bash\n', convertcmd])
    outcmd = 'bsub -J atm-ds008lvl1-%s -q PQ_nbc -e /scratch/PSB6351_2017/students/salo/crash/week08/frstlvl/ds008_R2_lvl1_err_%s -o /scratch/PSB6351_2017/students/salo/crash/week08/frstlvl/ds008_R2_lvl1_out_%s < %s' % (sid, sid, sid, script_file)
    os.system(outcmd)
    continue
