#!/bin/bash

#BSUB -J psb6351_dcm_convert
#BSUB -o /scratch/PSB6351_2017/week4/salo/crash/dcm_convert_out
#BSUB -e /scratch/PSB6351_2017/week4/salo/crash/dcm_convert_err

./dicomconvert2_GE.py -d /scratch/PSB6351_2017/dicoms/ -o /scratch/PSB6351_2017/week4/salo/data/ -f /scratch/PSB6351_2017/week4/salo/heuristic_shell.py -q PQ_nbc -c dcm2nii -s subj001

