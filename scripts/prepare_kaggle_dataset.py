import os
import shutil
import subprocess
from nyul_normalise import normalise_image

def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def callmyfunction(mycmd):
    cmd = subprocess.Popen(mycmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
    stdoutput = cmd.communicate()[0].decode("utf-8") .strip('\n')
    print(stdoutput)
    return stdoutput

def getResolution(image_full_name):
    import SimpleITK as sitk
    import numpy as np
    image     = sitk.ReadImage(image_full_name)
    pixelsize = image.GetSpacing()
    return np.array(pixelsize)

lvsc_challenge_dir = '/vol/biomedic2/oo2113/dataset/kaggle_atlas'
training_dir = os.path.join(lvsc_challenge_dir, 'testing/')
validation_dir = os.path.join(lvsc_challenge_dir, 'testing/')
target_dir = '/vol/biomedic2/oo2113/dataset/kaggle_structured'; mkdirfun(target_dir)

for directory, split in zip([training_dir, validation_dir], ['train', 'validation']):
    for category, interp_type in zip(['image','label'], ['-linear','-nn']):
        source_directory     = os.path.join(directory, category)
        target_split_dir     = os.path.join(target_dir, split); mkdirfun(target_split_dir)
        target_split_img_dir = os.path.join(target_split_dir, category); mkdirfun(target_split_img_dir)
        source_data = sorted(next(os.walk(source_directory))[2])

        for subject in source_data:
            subject_id = subject.split('_')[0]
            phase_id = subject.split('_')[-1]
            source_img_name = os.path.join(directory,category,subject)
            target_img_name = os.path.join(target_split_img_dir,subject_id+'_image_'+phase_id)
            shutil.copy(source_img_name, target_img_name)

            # resample the inplane resolution
            z_res = getResolution(target_img_name)[2]
            callmyfunction('resample {0} {0} {1} -size 1.82 1.82 {2}'.format(target_img_name, interp_type, z_res))

            # intensity histogram matching (Nyul's method)
            if category == 'image':
                print(target_img_name)
                normalise_image(target_img_name, target_img_name)