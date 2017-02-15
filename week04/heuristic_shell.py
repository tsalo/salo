import os

def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return (template, outtype, annotation_classes)

def infotodict(seqinfo):
    """Heuristic evaluator for determining which runs belong where
    
    allowed template fields - follow python string module: 
    
    item: index within category 
    subject: participant id 
    seqitem: run number during scanning
    subindex: sub index within group
    """
    
    dwi = create_key('dmri/dwi_{item:03d}', outtype=('dicom', 'nii.gz'))
    t1 = create_key('anatomy/T1_{item:03d}', outtype=('dicom', 'nii.gz'))
    bold = create_key('bold/bold_{item:03d}/bold', outtype=('dicom', 'nii.gz'))
    info = {dwi: [], t1: [], bold: []}
    last_run = len(seqinfo)
    for s in seqinfo:
        x,y,sl,nt = (s[6], s[7], s[8], s[9])
        if s[12].startswith('3D_T1'):
            info[t1].append(s[2])
        elif 'TRD' in s[12]:
            info[bold].append(s[2])
            last_run = s[2]
        elif 'DTI' in s[12]:
            info[dwi].append(s[2])
        else:
            pass
    return info
