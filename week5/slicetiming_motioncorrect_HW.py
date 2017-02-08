#!/usr/bin/env python

import os
import numpy as np
from nibabel import load
import nipype.interfaces.fsl as fsl
import nipype.interfaces.nipy as nipy
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util  
import nipype.interfaces.afni as afni


def pickfirst(func):
    """
    Return first item in list. Return whole input if not list.
    """
    if isinstance(func, list):
        return func[0]
    else:
        return func


def pickvol(filenames, fileidx, which):
    """
    """
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(np.ceil(load(filenames[fileidx]).get_shape()[3]/2))
    else:
        raise Exception('unknown value for volume selection : %s'%which)
    return idx


def get_subs(subject_id, mri_files):
    """
    """
    subs = []
    subs.append(('_subject_id_%s/' %subject_id, ''))
    for i, mri_file in enumerate(mri_files):
        subs.append(('_motion_sltime_correct%d/' %i, ''))
        subs.append(('_motion_correct%d/' %i, ''))
    return subs


def calc_slicetimes(filenames, TR):
    """
    TODO: Validate slice order assumption (interleaved ascending) using dicoms.
    Create slice order list from nifti image.
    Requires knowledge of slice timing (e.g., interleaved ascending).
    """
    all_slice_times = []
    for f in filenames:
       n_slices = load(f).get_shape()[2]
       slice_order = range(0, n_slices, 2) + range(1, n_slices, 2)
       slice_order = np.argsort(slice_order)
       slice_times = (slice_order * (TR / n_slices)).tolist()
       all_slice_times.append(slice_times)

    return all_slice_times

proj_dir = '/scratch/PSB6351/students/salo/'
work_dir = 'FILL IN WITH APPROPRIATE FILE PATH'
sink_dir = '/scratch/PSB6351/students/salo/week5/'

sids = ['sub-01'] # ALWAYS TEST WITH ONE PARTICIPANT AND THEN RUN ON FULL SAMPLE
#sids = ['sub-02', 'sub-03', 'sub-04', 'sub-05',
#        'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
#        'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15']

# Workflow
motcor_sltimes_wf = pe.Workflow('motcor_sltimes_wf')
motcor_sltimes_wf.base_dir = work_dir

# Define the outputs for the workflow
# FEEL FREE TO CHANGE WHAT YOU WANT TO SAVE. 
# AT A MINIMUM MUST INCLUDE REFERENCE EPI, MOTION CORRECTION PARAMETERS,
# AND EPI SCANS THAT HAVE BEEN MOTION CORRECTED AND SLICE TIME CORRECTED
output_fields = ['reference', 'motion_parameters',
                 'motion_sltime_corrected_files']
outputspec = pe.Node(util.IdentityInterface(fields=output_fields),
                     name='outputspec')


# Node: subject_iterable
subj_iterable = pe.Node(util.IdentityInterface(fields=['subject_id'],
                                               mandatory_inputs=True),
                        name='subj_iterable')
subj_iterable.iterables = ('subject_id', sids)


info = dict(mri_files=[['subject_id']])

# Create a datasource node to get the mri files
datasource = pe.Node(nio.DataGrabber(infields=['subject_id'], outfields=info.keys()), name='datasource')
datasource.inputs.template = '*'
datasource.inputs.base_directory = os.path.abspath(proj_dir)
datasource.inputs.field_template = dict(mri_files='FILL IN WITH WAY TO FIND FILES USING A DATAGRABBER')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True
datasource.inputs.ignore_exception = False
datasource.inputs.raise_on_empty = True
motcor_sltimes_wf.connect(subj_iterable, 'subject_id', datasource, 'subject_id')

# Create a Function node to rename output files with something more meaningful
getsubs = pe.Node(util.Function(input_names=['subject_id', 'mri_files'],
                                output_names=['subs'],
                                function=get_subs),
                  name='getsubs')
getsubs.inputs.ignore_exception = False
motcor_sltimes_wf.connect(subj_iterable, 'subject_id', getsubs, 'subject_id')
motcor_sltimes_wf.connect(datasource, 'mri_files', getsubs, 'mri_files')

# Extract the first volume of the first run as the reference 
extractref = pe.Node(fsl.ExtractROI(t_size=1),
                     iterfield=['in_file'],
                     name='extractref')
motcor_sltimes_wf.connect(datasource, ('mri_files', pickfirst),
                          extractref, 'in_file')
motcor_sltimes_wf.connect(datasource, ('mri_files', pickvol, 0, 'middle'),
                          extractref, 't_min')
motcor_sltimes_wf.connect('FOLLOW SYNTAX ABOVE TO CONNECT THIS NODE TO OUTPUTSPEC TO SAVE REFERENCE EPI VOLUME')

# NOTE: Committing to NIPY
# Simultaneous motion and slice timing correction with Nipy algorithm
motion_sltime_correct = pe.MapNode(nipy.SpaceTimeRealigner(),
                                   name='motion_sltime_correct',
                                   iterfield = ['in_file', 'slice_times'])
motion_sltime_correct.inputs.tr = 2.
motion_sltime_correct.inputs.slice_info = 2
motion_sltime_correct.plugin_args = {'bsub_args': '-n %s' %os.environ['MKL_NUM_THREADS']}
motion_sltime_correct.plugin_args = {'bsub_args': '-R "span[hosts=1]"'}
motcor_sltimes_wf.connect('FOLLOW SYNTAX TO CONNECT APPROPRIATE NODES TO INPUT IN_FILE')
motcor_sltimes_wf.connect(datasource, ('mri_files', calc_slicetimes, 2.),
                          motion_sltime_correct, 'slice_times') # ARE THESE APPROPRIATE?  HOW DO WE FIND OUT?
motcor_sltimes_wf.connect('WRITE APPROPRIATE SYNTAX TO CONNECT THIS NODE TO OUTPUT TO SAVE MOTION_PARAMETER FILES')
motcor_sltimes_wf.connect('WRITE APPROPRIATE SYNTAX TO CONNECT THIS NODE TO OUTPUT TO SAVE MOTION AND SLICE TIME CORRECTED EPI FILES')

# Save the relevant data into an output directory
datasink = pe.Node(nio.DataSink(), name='datasink')
datasink.inputs.base_directory = sink_dir
motcor_sltimes_wf.connect(subj_iterable, 'subject_id', datasink, 'container')
motcor_sltimes_wf.connect(outputspec, 'reference', datasink, 'ref')
motcor_sltimes_wf.connect(outputspec, 'motion_parameters', datasink, 'motion')
motcor_sltimes_wf.connect(outputspec, 'motion_sltime_corrected_files', datasink, 'func')
motcor_sltimes_wf.connect(getsubs, 'subs', datasink, 'substitutions')

# Run things and write crash files if necessary
motcor_sltimes_wf.config['execution']['crashdump_dir'] = '/scratch/PSB6351_2017/students/salo/crash/week5_nipype_crash'
motcor_sltimes_wf.base_dir = work_dir
motcor_sltimes_wf.run(plugin='LSF', plugin_args={'bsub_args': '-q PQ_nbc'})

# To view crash results: nipype_display_crash [crashdump_dir]/[crash_file].pklz
