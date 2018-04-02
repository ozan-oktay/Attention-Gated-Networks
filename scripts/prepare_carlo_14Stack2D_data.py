# Author: Ozan Oktay
# Date: January 18th, 2018

import os, glob
import subprocess
import shutil

def mkdirfunc(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def callmyfunction(mycmd):
    cmd = subprocess.Popen(mycmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
    stdoutput = cmd.communicate()[0].decode("utf-8") .strip('\n')
    print(stdoutput)
    return stdoutput

dcmdirectory = '/vol/biomedic2/oo2113/dataset/hammersmith_dataset/14_2DStack_data/'
subjectpaths = [name for name in os.listdir(dcmdirectory) if os.path.isdir(os.path.join(dcmdirectory,name))]
nifti_outdir = '/vol/bitbucket/oo2113/14_2DStack_nifti_data/'; mkdirfunc(nifti_outdir)
tmp_dir      = '/vol/bitbucket/oo2113/tmp_nifti/'; mkdirfunc(tmp_dir)

for subjectpath in sorted(subjectpaths):

    # clean the tmp dir
    shutil.rmtree(tmp_dir); mkdirfunc(tmp_dir)

    # convert the images
    subjectname = subjectpath.split('_')[0]
    subjectfullpath = os.path.join(dcmdirectory,subjectpath)
    print('processing: ', subjectname)
    callmyfunction('/vol/medic02/users/oo2113/Build/other/dcm2nii '
                   '-o {0} {1}/*dcm '.format(tmp_dir, subjectfullpath))

    # move the images
    nifti_files = glob.glob(os.path.join(tmp_dir,'*.nii.gz'))
    if len(nifti_files) > 3: continue
    for loop_id, nifti_file in enumerate(nifti_files,1):
        new_path = os.path.join(nifti_outdir, subjectname + '_' + str(loop_id) + '.nii.gz')
        shutil.move(nifti_file, new_path)