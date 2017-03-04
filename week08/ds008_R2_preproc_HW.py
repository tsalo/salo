#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os.path import join, abspath, dirname
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as fsl
import nipype.interfaces.nipy as nipy
import nipype.interfaces.afni as afni
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.utility as util  
import nipype.algorithms.confounds as conf
import nipype.algorithms.rapidart as ra

import numpy as np
import nibabel as nb
import shutil

imports = ['import os',
           'import nibabel as nb',
           'import numpy as np',
           'import scipy as sp',
           'from nipype.utils.filemanip import filename_to_list, list_to_filename, split_filename',
           'from scipy.special import legendre'
           ]


def pickfirst(func):
    if isinstance(func, list):
        return func[0]
    else:
        return func


def pickvol(filenames, fileidx, which):
    from nibabel import load
    import numpy as np
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(np.ceil(load(filenames[fileidx]).get_shape()[3]/2))
    else:
        raise Exception('unknown value for volume selection : %s'%which)
    return idx


def get_subs(subject_id, mri_files):
    subs = []
    subs.append(('_subject_id_%s/' %subject_id, ''))
    for i, mri_file in enumerate(mri_files):
        subs.append(('_motion_correct%d/' %i, ''))
        subs.append(('_art%d/' %i, ''))
        subs.append(('_susan_mask%d/' %i, ''))
    return subs


def motion_regressors(motion_params, order=0, derivatives=1):
    """Compute motion regressors up to given order and derivative

    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), 'motion_regressor%02d.txt' % idx)
        np.savetxt(filename, out_params2, fmt=str('%.5f'))
        out_files.append(filename)
    return out_files


def build_filter(motion_params, comp_norm, outliers, detrend_poly=None):
    """Builds a regressor set comprising motion parameters, composite norm and
    outliers

    The outliers are added as a single time point column for each outlier


    Parameters
    ----------

    motion_params: a text file containing motion parameters and its derivatives
    comp_norm: a text file containing the composite norm
    outliers: a text file containing 0-based outlier indices
    detrend_poly: number of polynomials to add to detrend

    Returns
    -------
    components_file: a text file containing all the regressors
    """
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        norm_val = np.genfromtxt(filename_to_list(comp_norm)[idx])
        out_params = np.hstack((params, norm_val[:, None]))
        if detrend_poly:
            timepoints = out_params.shape[0]
            X = np.ones((timepoints, 1))
            for i in range(detrend_poly):
                X = np.hstack((X, legendre(
                    i + 1)(np.linspace(-1, 1, timepoints))[:, None]))
            out_params = np.hstack((out_params, X))
        try:
            outlier_val = np.genfromtxt(filename_to_list(outliers)[idx])
        except IOError:
            outlier_val = np.empty((0))
        for index in np.atleast_1d(outlier_val):
            outlier_vector = np.zeros((out_params.shape[0], 1))
            outlier_vector[index] = 1
            out_params = np.hstack((out_params, outlier_vector))
        filename = os.path.join(os.getcwd(), 'filter_regressor%02d.txt' % idx)
        np.savetxt(filename, out_params, fmt=str('%.5f'))
        out_files.append(filename)
    return out_files


def getbtthresh(medianvals):
    """Get the brightness threshold for SUSAN."""
    return [0.75*val for val in medianvals]


def getusans(inlist):
    """Return the usans at the right threshold."""
    return [[tuple([val[0], 0.75*val[1]])] for val in inlist]


def get_aparc_aseg(files):
    for name in files:
        if 'aparc+aseg' in name:
            return name
    raise ValueError('aparc+aseg.mgz not found')


class_dir = '/scratch/PSB6351_2017/'
data_dir = join(class_dir, 'ds008_R2.0.0/')
subjects_dir = join(data_dir, 'surfaces/')
salo_dir = join(class_dir, 'students/salo/')
diagram_dir = dirname(__file__)
work_dir = join(salo_dir, 'working/week08/')
out_dir = join(salo_dir, 'data/func/')
err_dir = join(salo_dir, 'crash/week08/')

sids = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05',
        'sub-06', 'sub-07', 'sub-09', 'sub-10',
        'sub-11', 'sub-12', 'sub-13', 'sub-14', 'sub-15']

# Workflow
preproc_wf = pe.Workflow('preproc_wf')
preproc_wf.base_dir = work_dir

# Define the outputs for the workflow
output_fields = ['reference',
                 'mask_file',
                 'motion_parameters',
                 'motion_parameters_plusDerivs',
                 'motionandoutlier_noise_file',
                 'motion_corrected_files',
                 'afni_coreg_xfm',
                 'reg_file',
                 'reg_cost',
                 'reg_fsl_file',
                 'artnorm_files',
                 'artoutlier_files',
                 'artdisplacement_files',
                 'smoothed_files']

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
datasource.inputs.template = '*'
datasource.inputs.base_directory = abspath(data_dir)
datasource.inputs.field_template = dict(mri_files='%s/func/*stopsignal*_bold.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True
datasource.inputs.ignore_exception = False
datasource.inputs.raise_on_empty = True
preproc_wf.connect(subj_iterable, 'subject_id', datasource, 'subject_id')

# Create a Function node to rename output files with something more meaningful
getsubs = pe.Node(util.Function(input_names=['subject_id', 'mri_files'],
                                output_names=['subs'],
                                function=get_subs),
                  name='getsubs')
getsubs.inputs.ignore_exception = False
preproc_wf.connect(subj_iterable, 'subject_id', getsubs, 'subject_id')
preproc_wf.connect(datasource, 'mri_files', getsubs, 'mri_files')

# Extract the first volume of the first run as the reference 
extractref = pe.Node(fsl.ExtractROI(t_size=1),
                     iterfield=['in_file'],
                     name = 'extractref')
preproc_wf.connect(datasource, ('mri_files', pickfirst),
                   extractref, 'in_file')
preproc_wf.connect(datasource, ('mri_files', pickvol, 0, 'middle'),
                   extractref, 't_min')
preproc_wf.connect(extractref, 'roi_file', outputspec, 'reference')

# Motion correction with Nipy algorithm
motion_correct = pe.Node(nipy.SpaceTimeRealigner(),
                         name='motion_correct')
motion_correct.plugin_args = {'bsub_args': '-n 12'}
motion_correct.plugin_args = {'bsub_args': '-R "span[hosts=1]"'}
preproc_wf.connect(datasource, 'mri_files',
                   motion_correct, 'in_file')
preproc_wf.connect(motion_correct, 'par_file',
                   outputspec, 'motion_parameters')
preproc_wf.connect(motion_correct, 'out_file',
                   outputspec, 'motion_corrected_files')

# Calculate the transformation matrix from EPI space to FreeSurfer space
# using the BBRegister command
coregister = pe.Node(fs.BBRegister(subjects_dir=subjects_dir,
                                   contrast_type='t2',
                                   init='fsl',
                                   out_fsl_file=True),
                     name='coregister')
preproc_wf.connect(subj_iterable, 'subject_id', coregister, 'subject_id')
preproc_wf.connect(motion_correct, ('out_file', pickfirst),
                   coregister, 'source_file')
preproc_wf.connect(coregister, 'out_reg_file', outputspec, 'reg_file')
preproc_wf.connect(coregister, 'out_fsl_file', outputspec, 'fsl_reg_file')
preproc_wf.connect(coregister, 'min_cost_file', outputspec, 'reg_cost')

# Register a source file to fs space
fssource = pe.Node(nio.FreeSurferSource(subjects_dir=subjects_dir), name='fssource')
preproc_wf.connect(subj_iterable, 'subject_id', fssource, 'subject_id')

# Extract aparc+aseg brain mask and binarize
fs_threshold = pe.Node(fs.Binarize(min=0.5, out_type='nii'),
                       name ='fs_threshold')
preproc_wf.connect(fssource, ('aparc_aseg', get_aparc_aseg),
                   fs_threshold, 'in_file')

# Transform the binarized aparc+aseg file to the 1st volume of 1st run space
fs_voltransform = pe.MapNode(fs.ApplyVolTransform(inverse=True, subjects_dir=subjects_dir),
                             iterfield=['source_file', 'reg_file'],
                             name='fs_transform')
preproc_wf.connect(extractref, 'roi_file', fs_voltransform, 'source_file')
preproc_wf.connect(coregister, 'out_reg_file', fs_voltransform, 'reg_file')
preproc_wf.connect(fs_threshold, 'binary_file', fs_voltransform, 'target_file')

# Dilate the binarized mask by 1 voxel that is now in the EPI space
fs_threshold2 = pe.MapNode(fs.Binarize(min=0.5, out_type='nii', dilate=1),
                           iterfield=['in_file'],
                           name='fs_threshold2')
preproc_wf.connect(fs_voltransform, 'transformed_file',
                   fs_threshold2, 'in_file')
preproc_wf.connect(fs_threshold2, 'binary_file', outputspec, 'mask_file')

# Mask the functional runs with the extracted mask
maskfunc = pe.MapNode(fsl.ImageMaths(suffix='_bet',
                                     op_string='-mas'),
                      iterfield=['in_file'],
                      name = 'maskfunc')
preproc_wf.connect(motion_correct, 'out_file', maskfunc, 'in_file')
preproc_wf.connect(fs_threshold2, ('binary_file', pickfirst),
                   maskfunc, 'in_file2')

# Use RapidART to detect motion/intensity outliers
art = pe.MapNode(ra.ArtifactDetect(use_differences=[True, False],
                                   use_norm=True,
                                   zintensity_threshold=3,
                                   norm_threshold=1,
                                   bound_by_brainmask=True,
                                   mask_type='file'),
                 iterfield=['realignment_parameters', 'realigned_files'],
                 name='art')
art.inputs.parameter_source = 'NiPy'
preproc_wf.connect(motion_correct, 'par_file',
                   art, 'realignment_parameters')
preproc_wf.connect(motion_correct, 'out_file',
                   art, 'realigned_files')
preproc_wf.connect(fs_threshold2, ('binary_file', pickfirst),
                   art, 'mask_file')
preproc_wf.connect(art, 'norm_files', outputspec, 'artnorm_files')
preproc_wf.connect(art, 'outlier_files', outputspec, 'artoutlier_files')
preproc_wf.connect(art, 'displacement_files',
                   outputspec, 'artdisplacement_files')

# Compute motion regressors (save file with 1st and 2nd derivatives)
motreg = pe.Node(util.Function(input_names=['motion_params', 'order',
                                            'derivatives'],
                               output_names=['out_files'],
                               function=motion_regressors,
                               imports=imports),
                 name='getmotionregress')
preproc_wf.connect(motion_correct, 'par_file', motreg, 'motion_params')
preproc_wf.connect(motreg, 'out_files',
                   outputspec, 'motion_parameters_plusDerivs')

# Create a filter text file to remove motion (+ derivatives), art confounds,
# and 1st, 2nd, and 3rd order legendre polynomials.
createfilter = pe.Node(util.Function(input_names=['motion_params', 'comp_norm',
                                                  'outliers', 'detrend_poly'],
                                     output_names=['out_files'],
                                     function=build_filter,
                                     imports=imports),
                        name='makemotionbasedfilter')
createfilter.inputs.detrend_poly = 3
preproc_wf.connect(motreg, 'out_files', createfilter, 'motion_params')
preproc_wf.connect(art, 'norm_files', createfilter, 'comp_norm')
preproc_wf.connect(art, 'outlier_files', createfilter, 'outliers')
preproc_wf.connect(createfilter, 'out_files',
                   outputspec, 'motionandoutlier_noise_file')

### SUSAN Smoothing
# Smooth each run using SUSAN with the brightness threshold set to 75%
# of the median value for each run and a mask constituting the mean
# functional
smooth_median = pe.MapNode(fsl.ImageStats(op_string='-k %s -p 50'),
                           iterfield = ['in_file'],
                           name='susan_smooth_median')
preproc_wf.connect(maskfunc, 'out_file', smooth_median, 'in_file')
preproc_wf.connect(fs_threshold2, ('binary_file', pickfirst),
                   smooth_median, 'mask_file')

smooth_meanfunc = pe.MapNode(fsl.ImageMaths(op_string='-Tmean',
                                            suffix='_mean'),
                                iterfield=['in_file'],
                                name='susan_smooth_meanfunc')
preproc_wf.connect(maskfunc, 'out_file', smooth_meanfunc, 'in_file')

smooth_merge = pe.Node(util.Merge(2, axis='hstack'),
                       name='susan_smooth_merge')
preproc_wf.connect(smooth_meanfunc, 'out_file', smooth_merge, 'in1')
preproc_wf.connect(smooth_median, 'out_stat', smooth_merge, 'in2')

susan_smooth = pe.MapNode(fsl.SUSAN(),
                          iterfield=['in_file', 'brightness_threshold',
                                     'usans'],
                          name='susan_smooth')
susan_smooth.inputs.fwhm = 6.
preproc_wf.connect(maskfunc, 'out_file', susan_smooth, 'in_file')
preproc_wf.connect(smooth_median, ('out_stat', getbtthresh),
                   susan_smooth, 'brightness_threshold')
preproc_wf.connect(smooth_merge, ('out', getusans),
                   susan_smooth, 'usans')

# Mask the smoothed data with the dilated mask
maskfunc2 = pe.MapNode(fsl.ImageMaths(suffix='_mask',
                                      op_string='-mas'),
                       iterfield=['in_file'],
                       name='susan_mask')
preproc_wf.connect(susan_smooth, 'smoothed_file', maskfunc2, 'in_file')
preproc_wf.connect(fs_threshold2, ('binary_file', pickfirst),
                   maskfunc2, 'in_file2')
preproc_wf.connect(maskfunc2, 'out_file',
                   outputspec, 'smoothed_files')

# Save the relevant data into an output directory
datasink = pe.Node(nio.DataSink(), name='datasink')
datasink.inputs.base_directory = out_dir
preproc_wf.connect(outputspec, 'smoothed_files',
                   datasink, 'preproc.func.smoothed')

preproc_wf.connect(subj_iterable, 'subject_id', datasink, 'container')
preproc_wf.connect(outputspec, 'reference', datasink, 'preproc.ref')
preproc_wf.connect(outputspec, 'motion_parameters',
                   datasink, 'preproc.motion')
preproc_wf.connect(outputspec, 'motion_corrected_files',
                   datasink, 'preproc.func.realigned')
preproc_wf.connect(getsubs, 'subs', datasink, 'substitutions')
preproc_wf.connect(outputspec, 'reg_file', datasink, 'preproc.bbreg.@reg')
preproc_wf.connect(outputspec, 'reg_cost', datasink, 'preproc.bbreg.@cost')
preproc_wf.connect(outputspec, 'reg_fsl_file',
                   datasink, 'preproc.bbreg.@regfsl')
preproc_wf.connect(outputspec, 'artnorm_files',
                   datasink, 'preproc.art.@norm_files')
preproc_wf.connect(outputspec, 'artoutlier_files',
                   datasink, 'preproc.art.@outlier_files')
preproc_wf.connect(outputspec, 'artdisplacement_files',
                   datasink, 'preproc.art.@displacement_files')
preproc_wf.connect(outputspec, 'motion_parameters_plusDerivs',
                   datasink, 'preproc.noise.@motionplusDerivs')
preproc_wf.connect(outputspec, 'motionandoutlier_noise_file',
                   datasink, 'preproc.noise.@motionplusoutliers')
preproc_wf.connect(outputspec, 'mask_file', datasink, 'preproc.ref.@mask')

# Create and copy graphs to output directory for easy access.
preproc_wf.write_graph(graph2use='exec')

shutil.copy(join(preproc_wf.base_dir, preproc_wf.name, 'graph_detailed.dot.png'),
            join(diagram_dir, 'preproc_graph_detailed.png'))
shutil.copy(join(preproc_wf.base_dir, preproc_wf.name, 'graph.dot.png'),
            join(diagram_dir, 'preproc_graph_basic.png'))

# Run things and write crash files if necessary
preproc_wf.config['execution']['crashdump_dir'] = err_dir
preproc_wf.base_dir = work_dir
preproc_wf.run(plugin='LSF', plugin_args={'bsub_args': '-q PQ_nbc'})
