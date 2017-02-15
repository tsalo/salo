#!/usr/bin/env python

import os
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
    import numpy as np
    from nibabel import load
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


proj_dir = '/scratch/PSB6351_2017/ds008_R2.0.0/'
subjects_dir = '/scratch/PSB6351_2017/ds008_R2.0.0/surfaces/'
work_dir = '/scratch/PSB6351_2017/students/salo/working/'
sink_dir = '/scratch/PSB6351_2017/students/salo/data/'
err_dir = '/scratch/PSB6351_2017/students/salo/crash/week06/'

sids = ['sub-01']#, 'sub-02', 'sub-03', 'sub-04', 'sub-05',
#        'sub-06', 'sub-07', 'sub-09', 'sub-10', 'sub-11',
#        'sub-12', 'sub-13', 'sub-14', 'sub-15']

##
# Workflow
preproc_wf = pe.Workflow('preproc_wf')
preproc_wf.base_dir = work_dir

# Define the outputs for the workflow
output_fields = ['reference',
                 'motion_parameters',
                 'motion_corrected_files',
                 'reg_file',
                 'reg_cost',
                 'fsl_reg_file']
outputspec = pe.Node(util.IdentityInterface(fields=output_fields),
                     name='outputspec')

# Node: subject_iterable
subj_iterable = pe.Node(util.IdentityInterface(fields=['subject_id'],
                                               mandatory_inputs=True),
                        name='subj_iterable')
subj_iterable.iterables = ('subject_id', sids)

info = dict(mri_files=[['subject_id']])

# Create a datasource node to get the mri files
datasource = pe.Node(nio.DataGrabber(infields=['subject_id'],
                                     outfields=info.keys()),
                     name='datasource')
datasource.inputs.template = '*_bold.nii.gz'
datasource.inputs.base_directory = os.path.abspath(proj_dir)
datasource.inputs.field_template = dict(mri_files='%s/func/*_bold.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True
datasource.inputs.ignore_exception = False
datasource.inputs.raise_on_empty = True
preproc_wf.connect(subj_iterable, 'subject_id',
                   datasource, 'subject_id')

# Create a Function node to rename output files with something more meaningful
getsubs = pe.Node(util.Function(input_names=['subject_id', 'mri_files'],
                                output_names=['subs'],
                                function=get_subs),
                  name='getsubs')
getsubs.inputs.ignore_exception = False
preproc_wf.connect(subj_iterable, 'subject_id',
                   getsubs, 'subject_id')
preproc_wf.connect(datasource, 'mri_files',
                   getsubs, 'mri_files')

# Extract the first volume of the first run as the reference 
extractref = pe.Node(fsl.ExtractROI(t_size=1, t_min=0),
                     iterfield=['in_file'],
                     name='extractref')
preproc_wf.connect(datasource, ('mri_files', pickfirst),
                   extractref, 'in_file')
preproc_wf.connect(extractref, 'roi_file',
                   outputspec, 'reference')

# NOTE: Committing to NIPY
# Simultaneous motion and slice timing correction with Nipy algorithm
motion_correct = pe.MapNode(nipy.SpaceTimeRealigner(),
                            name='motion_correct',
                            iterfield=['in_file'])
motion_correct.plugin_args = {'bsub_args': '-n 12'}
motion_correct.plugin_args = {'bsub_args': '-R "span[hosts=1]"'}
preproc_wf.connect(datasource, 'mri_files',
                   motion_correct, 'in_file')
preproc_wf.connect(motion_correct, 'par_file',
                   outputspec, 'motion_parameters')
preproc_wf.connect(motion_correct, 'out_file',
                   outputspec, 'motion_corrected_files')


coregister = pe.MapNode(fs.BBRegister(subjects_dir=subjects_dir,
                                      contrast_type='t1',
                                      init='header',
                                      out_fsl_file=True),
                        name='coregister',
                        iterfield=['source_file'])
preproc_wf.connect(subj_iterable, 'subject_id', coregister, 'subject_id')
preproc_wf.connect(motion_correct, 'out_file', coregister, 'source_file')

preproc_wf.connect(coregister, 'out_reg_file', outputspec, 'reg_file')
preproc_wf.connect(coregister, 'out_fsl_file', outputspec, 'fsl_reg_file')
preproc_wf.connect(coregister, 'min_cost_file', outputspec, 'reg_cost')

# Save the relevant data into an output directory
# Save the relevant data into an output directory
#rename_reg_cost = pe.MapNode(util.Rename(format_string='%(seq)s.mincost',
#                                         parse_string='.*corr_(?P<seq>.*)_.*'),
#                             name='rename_reg_cost', iterfield=['in_file'])

datasink = pe.Node(nio.DataSink(), name='datasink')
datasink.inputs.base_directory = sink_dir
preproc_wf.connect(subj_iterable, 'subject_id', datasink, 'container')
preproc_wf.connect(outputspec, 'reference', datasink, 'preproc.ref')
preproc_wf.connect(outputspec, 'motion_parameters', datasink, 'preproc.motion')
preproc_wf.connect(outputspec, 'motion_corrected_files', datasink, 'preproc.func')
preproc_wf.connect(outputspec, 'reg_file', datasink, 'preproc.reg_file')
preproc_wf.connect(outputspec, 'reg_cost', datasink, 'preproc.reg_cost')
#preproc_wf.connect(outputspec, 'reg_cost', rename_reg_cost, 'in_file')
#preproc_wf.connect(rename_reg_cost, 'out_file', datasink, 'preproc.reg_cost')

preproc_wf.connect(outputspec, 'fsl_reg_file', datasink, 'preproc.fsl_reg_file')
preproc_wf.connect(getsubs, 'subs', datasink, 'substitutions')

# Run things and write crash files if necessary
preproc_wf.config['execution']['crashdump_dir'] = err_dir
preproc_wf.base_dir = work_dir

preproc_wf.write_graph(graph2use='flat')
preproc_wf.run(plugin='LSF', plugin_args={'bsub_args': '-q PQ_nbc'})
