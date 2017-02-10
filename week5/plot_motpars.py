# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 12:37:08 2017

@author: tsalo006
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from glob import glob
from os.path import join, basename, splitext

# Directions per http://web.mit.edu/fsl_v5.0.8/fsl/doc/wiki/POSSUM(2f)UserGuide.html
rot_dirs = ['Pitch', 'Roll', 'Yaw']
trans_dirs = ['X', 'Y', 'Z']
colors = ['r', 'b', 'g']

data_dir = '/scratch/PSB6351_2017/students/salo/data/'

subjects = [basename(f) for f in glob(join(data_dir, '*'))]
for s in subjects[:1]:
    print(s)
    fig.suptitle('Movement: {0}'.format(s))
    
    motpar_files = sorted([basename(f) for f in glob(join(data_dir, s, 'func/*.par'))])
    n_runs = len(motpar_files)
    fig, axarr = plt.subplots(n_runs, 2)
    
    for run in range(n_runs):
        run_name = motpar_files[run][:5]
        print('\t{0}'.format(run_name))
        dat = np.loadtxt(join(data_dir, s, 'func', motpar_files[run]))
        rot, trans = dat[:, :3], dat[:, 3:]
        
        for i, rot_dir in enumerate(rot_dirs):
            axarr[run, 0].plot(range(dat.shape[0]), rot[:, i], c=colors[i], label=rot_dir)

        axarr[run, 0].set_xlabel('TR')
        axarr[run, 0].set_ylabel('Movement (in radians)')
        
        for i, trans_dir in enumerate(trans_dirs):
            axarr[run, 1].plot(range(dat.shape[0]), trans[:, i], c=colors[i], label=trans_dir)

        axarr[run, 1].set_ylabel('Movement (in mm)')
        
    #fig.savefig("/Users/salo/NBCLab/brainmap-vs-neurosynth/figures/database-coverage.png", dpi=400)
    axarr[-1, 1].set_xlabel('TR')

    legend = ax.legend(loc=(.1, .66), frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')