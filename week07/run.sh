#!/bin/bash 
#---Number of core
#BSUB -n 12
#BSUB -R "span[ptile=12]"

#---Job's name in LSF system
#BSUB -J w7

#---Error file
#BSUB -eo err_w7

#---Output file
#BSUB -oo out_w7

#---LSF Queue name
#BSUB -q PQ_nbc

##########################################################
# Set up environmental variables.
##########################################################
export NPROCS=`echo $LSB_HOSTS | wc -w`
export OMP_NUM_THREADS=$NPROCS

. $MODULESHOME/../global/profile.modules

source /scratch/PSB6351_2017/set_psb6351_env.sh
python /scratch/PSB6351_2017/students/salo/week07/ds008_R2_preproc_HW.py
