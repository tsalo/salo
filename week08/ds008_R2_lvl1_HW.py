#!/usr/bin/env python

"""
================================================================
ds008_R2.0.0_fMRI: FSL
================================================================

A firstlevel workflow for openfMRI ds008_R2.0.0 task data.

This workflow makes use of:

- FSL

For example::

  python ds008_R2_lvl1.py -s sub-01
                          -o /scratch/PSB6351_2017/students/MYNAME/week8/output
                          -w /scratch/PSB6351_2017/students/MYNAME/week8/workdir

"""

import os
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.utility import Function
from nipype.utils.misc import getsource
from nipype.interfaces.io import DataGrabber
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl.model import Level1Design
from nipype.interfaces.fsl.model import FEATModel
from nipype.interfaces.fsl.model import FILMGLS
from nipype.interfaces.fsl.model import ContrastMgr
from nipype.interfaces.fsl.utils import ImageMaths
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Merge

# Functions
pop_lambda = lambda x : x[0]

def subjectinfo(subject_id):
    import os
    from nipype.interfaces.base import Bunch
    from copy import deepcopy
    import numpy as np
    
    vec_dir = '/scratch/PSB6351_2017/students/salo/data/EVs_RTamp/'
    runs = ['run-{0:02d}'.format(r+1) for r in range(3)]
    conditions = ['failed_stop', 'successful_stop', 'go', 'junk']
    
    output = []
    for run in runs:
        names, onsets, durations, amplitudes = [], [], [], []
        for cond in conditions:
            cond_name = '_'.join(cond.split(' '))
            cond_file = os.path.join(vec_dir, subject_id,
                                     '{0}.{1}.txt'.format(cond_name, run))
            if os.path.isfile(cond_file):
                names.append(cond)
                data = np.genfromtxt(cond_file)
                if len(data.shape) < 2:
                    data = data[None, :]
                onsets.append(map(float, data[:, 0]))
                durations.append(map(float, data[:, 1]))
                amplitudes.append(map(float, data[:, 2]))       

        output.append(Bunch(conditions=names,
                            onsets=deepcopy(onsets),
                            durations=deepcopy(durations),
                            amplitudes=deepcopy(amplitudes),
                            tmod=None,
                            pmod=None,
                            regressor_names=None,
                            regressors=None))
    return output


def get_contrasts(subject_id):
    contrasts = [['AllVsBase', 'T',
                  ['failed_stop', 'go', 'successful_stop', 'junk'],
                  [0.25, 0.25, 0.25, 0.25]],
                 ['SuccessfulStopVsGo', 'T', ['successful_stop', 'go'], [1., -1.]],
                 ['SuccessfulStopVsFailedStop', 'T', ['successful_stop', 'failed_stop'], [1., -1.]],
                 ['FailedStopVsSuccessfulStop', 'T', ['successful_stop', 'failed_stop'], [-1., 1.]],]
    return contrasts


def get_subs(subject_id, cons, info):
    subs = []
    runs = range(1, len(info)+1)
    for i, run in enumerate(runs):
        subs.append(('_modelestimate%d/'%i, '_run_%d_%02d_'%(i, run)))
        subs.append(('_modelgen%d/'%i, '_run_%d_%02d_'%(i, run)))
        subs.append(('_conestimate%d/'%i, '_run_%d_%02d_'%(i, run)))
        subs.append(('_estimate_model%d/'%i, 'run-%02d/'%(run)))
    
    for i, con in enumerate(cons):
        subs.append(('cope%d.'%(i+1), 'cope%02d_%s.'%(i+1,con[0])))
        subs.append(('varcope%d.'%(i+1), 'varcope%02d_%s.'%(i+1,con[0])))
        subs.append(('_z2pval%d/zstat%d'%(i, i+1), 'zstat%02d_%s'%(i+1,con[0])))
        subs.append(('tstat%d.'%(i+1), 'tstat%02d_%s.'%(i+1,con[0])))
        
    return subs


def motion_noise(subjinfo, files):
    """
    subjinfo: nRuns length list of vectors(?)
    files: nRuns(?) length list of nuisance files
    """
    import numpy as np
    
    motion_noise_params = []
    motion_noi_par_names = []
    if not isinstance(files, list):
        files = [files]
    
    if not isinstance(subjinfo, list):
        subjinfo = [subjinfo]
    
    for i_run, file_ in enumerate(files):
        curr_mot_noi_par_names = ['Pitch (rad)', 'Roll (rad)', 'Yaw (rad)',
                                  'Tx (mm)', 'Ty (mm)', 'Tz (mm)',
                                  'Pitch_1d', 'Roll_1d', 'Yaw_1d',
                                  'Tx_1d', 'Ty_1d', 'Tz_1d',
                                  'Norm (mm)', 'LG_1stOrd', 'LG_2ndOrd',
                                  'LG_3rdOrd', 'LG_4thOrd']
        mot_data = np.genfromtxt(file_)
        motion_noise_params.append([[]] * mot_data.shape[1])
        
        # Make standard names for any nuisance params outside the expected 17
        if mot_data.shape[1] > 17:
            for num_out in range(mot_data.shape[1] - 17):
                out_name = 'out_%s' %(num_out+1)
                curr_mot_noi_par_names.append(out_name)
        
        for param in range(mot_data.shape[1]):
            motion_noise_params[i_run][param] = mot_data[:, param].tolist()
        motion_noi_par_names.append(curr_mot_noi_par_names)
    
    for j_run, run in enumerate(subjinfo):
        if run.regressor_names is None:
            run.regressor_names = []
        
        if run.regressors is None:
            run.regressors = []
        
        for k_param, run_nuisance in enumerate(motion_noise_params[j_run]):
            run.regressor_names.append(motion_noi_par_names[j_run][k_param])
            run.regressors.append(run_nuisance)
    return subjinfo


def firstlevel_wf(subject_id, sink_directory, name='ds008_R2_frstlvl_wf'):
    
    frstlvl_wf = Workflow(name='frstlvl_wf')

    info = dict(task_mri_files=[['subject_id', 'stopsignal']],
                motion_noise_files=[['subject_id', 'filter_regressor']])

    # Create a Function node to define stimulus onsets, etc... for each subject
    subject_info = Node(Function(input_names=['subject_id'],
                                 output_names=['output'],
                                 function=subjectinfo),
                        name='subject_info')
    subject_info.inputs.ignore_exception = False
    subject_info.inputs.subject_id = subject_id

    # Create another Function node to define the contrasts for the experiment
    getcontrasts = Node(Function(input_names=['subject_id'],
                                 output_names=['contrasts'],
                                 function=get_contrasts),
                        name='getcontrasts')
    getcontrasts.inputs.ignore_exception = False
    getcontrasts.inputs.subject_id = subject_id

    # Create a Function node to substitute names of files created during pipeline
    getsubs = Node(Function(input_names=['subject_id', 'cons', 'info'],
                            output_names=['subs'],
                            function=get_subs),
                   name='getsubs')
    getsubs.inputs.ignore_exception = False
    getsubs.inputs.subject_id = subject_id
    frstlvl_wf.connect(subject_info, 'output', getsubs, 'info')
    frstlvl_wf.connect(getcontrasts, 'contrasts', getsubs, 'cons')
    
    # Create a datasource node to get the task_mri and motion-noise files
    datasource = Node(DataGrabber(infields=['subject_id'],
                                  outfields=info.keys()),
                      name='datasource')
    datasource.inputs.template = '*'
    datasource.inputs.subject_id = subject_id
    #datasource.inputs.base_directory = os.path.abspath('/scratch/PSB6351_2017/ds008_R2.0.0/preproc/')
    #datasource.inputs.field_template = dict(task_mri_files='%s/func/realigned/*%s*.nii.gz',
    #                                        motion_noise_files='%s/noise/%s*.txt') 
    datasource.inputs.base_directory = os.path.abspath('/scratch/PSB6351_2017/students/salo/data/preproc/')
    datasource.inputs.field_template = dict(task_mri_files='%s/preproc/func/smoothed/corr_*_task-%s_*_bold_bet_smooth_mask.nii.gz',
                                            motion_noise_files='%s/preproc/noise/%s*.txt')
    datasource.inputs.template_args = info
    datasource.inputs.sort_filelist = True
    datasource.inputs.ignore_exception = False
    datasource.inputs.raise_on_empty = True
    
    # Create a Function node to modify the motion and noise files to be single regressors
    motionnoise = Node(Function(input_names=['subjinfo', 'files'],
                                output_names=['subjinfo'],
                                function=motion_noise),
                       name='motionnoise')
    motionnoise.inputs.ignore_exception = False
    frstlvl_wf.connect(subject_info, 'output', motionnoise, 'subjinfo')
    frstlvl_wf.connect(datasource, 'motion_noise_files', motionnoise, 'files')

    # Create a specify model node
    specify_model = Node(SpecifyModel(), name='specify_model')
    specify_model.inputs.high_pass_filter_cutoff = 128.
    specify_model.inputs.ignore_exception = False
    specify_model.inputs.input_units = 'secs'
    specify_model.inputs.time_repetition = 2.
    frstlvl_wf.connect(datasource, 'task_mri_files', specify_model, 'functional_runs')
    frstlvl_wf.connect(motionnoise, 'subjinfo', specify_model, 'subject_info')

    # Create an InputSpec node for the modelfit node
    modelfit_inputspec = Node(IdentityInterface(fields=['session_info', 'interscan_interval',
                                                        'contrasts', 'film_threshold',
                                                        'functional_data', 'bases',
                                                        'model_serial_correlations'],
                                                mandatory_inputs=True),
                              name='modelfit_inputspec')
    modelfit_inputspec.inputs.bases = {'dgamma':{'derivs': False}}
    modelfit_inputspec.inputs.film_threshold = 0.0
    modelfit_inputspec.inputs.interscan_interval = 2.0
    modelfit_inputspec.inputs.model_serial_correlations = True
    frstlvl_wf.connect(datasource, 'task_mri_files',
                       modelfit_inputspec, 'functional_data')
    frstlvl_wf.connect(getcontrasts, 'contrasts',
                       modelfit_inputspec, 'contrasts')
    frstlvl_wf.connect(specify_model, 'session_info',
                       modelfit_inputspec, 'session_info')

    # Create a level1 design node
    level1_design = Node(Level1Design(), name='level1_design')
    level1_design.inputs.ignore_exception = False
    frstlvl_wf.connect(modelfit_inputspec, 'interscan_interval',
                       level1_design, 'interscan_interval')
    frstlvl_wf.connect(modelfit_inputspec, 'session_info',
                       level1_design, 'session_info')
    frstlvl_wf.connect(modelfit_inputspec, 'contrasts',
                       level1_design, 'contrasts')
    frstlvl_wf.connect(modelfit_inputspec, 'bases', level1_design, 'bases')
    frstlvl_wf.connect(modelfit_inputspec, 'model_serial_correlations',
                       level1_design, 'model_serial_correlations')

    # Create a MapNode to generate a model for each run
    generate_model = MapNode(FEATModel(),
                             iterfield=['fsf_file', 'ev_files'],
                             name='generate_model')
    generate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    generate_model.inputs.ignore_exception = False
    generate_model.inputs.output_type = 'NIFTI_GZ'
    generate_model.inputs.terminal_output = 'stream'
    frstlvl_wf.connect(level1_design, 'fsf_files',
                       generate_model, 'fsf_file')
    frstlvl_wf.connect(level1_design, 'ev_files',
                       generate_model, 'ev_files')

    # Create a MapNode to estimate the model using FILMGLS
    estimate_model = MapNode(FILMGLS(),
                             iterfield=['design_file', 'in_file', 'tcon_file'],
                             name='estimate_model')
    frstlvl_wf.connect(generate_model, 'design_file',
                       estimate_model, 'design_file')
    frstlvl_wf.connect(generate_model, 'con_file', estimate_model, 'tcon_file')
    frstlvl_wf.connect(modelfit_inputspec, 'functional_data',
                       estimate_model, 'in_file')

    # Create a merge node to merge the contrasts - necessary for fsl 5.0.7 and greater
    merge_contrasts = MapNode(Merge(2), iterfield=['in1'],
                              name='merge_contrasts')
    frstlvl_wf.connect(estimate_model, 'zstats', merge_contrasts, 'in1')

    # Create a MapNode to transform the z2pval
    z2pval = MapNode(ImageMaths(), iterfield=['in_file'], name='z2pval')
    z2pval.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    z2pval.inputs.ignore_exception = False
    z2pval.inputs.op_string = '-ztop'
    z2pval.inputs.output_type = 'NIFTI_GZ'
    z2pval.inputs.suffix = '_pval'
    z2pval.inputs.terminal_output = 'stream'
    frstlvl_wf.connect(merge_contrasts, ('out', pop_lambda), z2pval, 'in_file')

    # Create an outputspec node
    modelfit_outputspec = Node(IdentityInterface(fields=['copes', 'varcopes',
                                                         'dof_file', 'pfiles',
                                                         'parameter_estimates',
                                                         'zstats', 'design_image',
                                                         'design_file', 'design_cov',
                                                         'sigmasquareds'],
                                                 mandatory_inputs=True),
                               name='modelfit_outputspec')
    frstlvl_wf.connect(estimate_model, 'copes', modelfit_outputspec, 'copes')
    frstlvl_wf.connect(estimate_model, 'varcopes',
                       modelfit_outputspec, 'varcopes')
    frstlvl_wf.connect(merge_contrasts, 'out', modelfit_outputspec, 'zstats')
    frstlvl_wf.connect(z2pval, 'out_file', modelfit_outputspec, 'pfiles')
    frstlvl_wf.connect(generate_model, 'design_image',
                       modelfit_outputspec, 'design_image')
    frstlvl_wf.connect(generate_model, 'design_file',
                       modelfit_outputspec, 'design_file')
    frstlvl_wf.connect(generate_model, 'design_cov',
                       modelfit_outputspec, 'design_cov')
    frstlvl_wf.connect(estimate_model, 'param_estimates',
                       modelfit_outputspec, 'parameter_estimates')
    frstlvl_wf.connect(estimate_model, 'dof_file',
                       modelfit_outputspec, 'dof_file')
    frstlvl_wf.connect(estimate_model, 'sigmasquareds',
                       modelfit_outputspec, 'sigmasquareds')

    # Create a datasink node
    sinkd = Node(DataSink(), name='sinkd')
    sinkd.inputs.base_directory = sink_directory 
    sinkd.inputs.container = subject_id
    frstlvl_wf.connect(getsubs, 'subs', sinkd, 'substitutions')
    frstlvl_wf.connect(modelfit_outputspec, 'parameter_estimates',
                       sinkd, 'modelfit.estimates')
    frstlvl_wf.connect(modelfit_outputspec, 'sigmasquareds',
                       sinkd, 'modelfit.estimates.@sigsq')
    frstlvl_wf.connect(modelfit_outputspec, 'dof_file',
                       sinkd, 'modelfit.dofs')
    frstlvl_wf.connect(modelfit_outputspec, 'copes',
                       sinkd, 'modelfit.contrasts.@copes')
    frstlvl_wf.connect(modelfit_outputspec, 'varcopes',
                       sinkd, 'modelfit.contrasts.@varcopes')
    frstlvl_wf.connect(modelfit_outputspec, 'zstats',
                       sinkd, 'modelfit.contrasts.@zstats')
    frstlvl_wf.connect(modelfit_outputspec, 'design_image',
                       sinkd, 'modelfit.design')
    frstlvl_wf.connect(modelfit_outputspec, 'design_cov',
                       sinkd, 'modelfit.design.@cov')
    frstlvl_wf.connect(modelfit_outputspec, 'design_file',
                       sinkd, 'modelfit.design.@matrix')
    frstlvl_wf.connect(modelfit_outputspec, 'pfiles',
                       sinkd, 'modelfit.contrasts.@pstats')

    return frstlvl_wf


def create_frstlvl_workflow(args, name='ds008_R2_frstlvl'):
    """
    Creates the full workflow
    """
    kwargs = dict(subject_id=args.subject_id,
                  sink_directory=os.path.abspath(args.out_dir),
                  name=name)
    frstlvl_workflow = firstlevel_wf(**kwargs)
    return frstlvl_workflow


if __name__ == '__main__':
    from argparse import ArgumentParser
    import shutil
    from os.path import dirname, join
    diagram_dir = dirname(__file__)
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('-s', '--subject_id', dest='subject_id',
                        help='Current subject id', required=True)
    parser.add_argument('-o', '--output_dir', dest='out_dir',
                        help='Output directory base')
    parser.add_argument('-w', '--work_dir', dest='work_dir',
                        help='Working directory base')
    args = parser.parse_args()

    wf = create_frstlvl_workflow(args)

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.getcwd()

    wf.base_dir = os.path.join(work_dir, args.subject_id)
    wf.write_graph(graph2use='exec')
    
    shutil.copy(join(wf.base_dir, wf.name, 'graph_detailed.dot.png'),
                join(diagram_dir, 'frstlvl_graph_detailed.png'))
    shutil.copy(join(wf.base_dir, wf.name, 'graph.dot.png'),
                join(diagram_dir, 'frstlvl_graph_basic.png'))
    
    wf.config['execution']['crashdump_dir'] = '/scratch/PSB6351_2017/students/salo/crash/week08/'
    wf.run(plugin='LSF', plugin_args={'bsub_args': '-q PQ_nbc'})
