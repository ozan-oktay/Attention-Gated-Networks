import os
import csv
import subprocess
import random

def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def callmyfunction(mycmd):
    cmd = subprocess.Popen(mycmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
    stdoutput = cmd.communicate()[0].decode("utf-8") .strip('\n')
    print(stdoutput)
    return stdoutput


# Randomly split the training set into 80 for train and 20 for test
data_path = '/vol/biomedic2/oo2113/dataset/ACDC_data_structured'
dest_path = '/vol/biomedic2/oo2113/dataset/ACDC_data_structured_split'; mkdirfun(dest_path)
dest_path_train = os.path.join(dest_path, 'train'); mkdirfun(dest_path_train)
dest_path_valid = os.path.join(dest_path, 'validation'); mkdirfun(dest_path_valid)

data_img_path = os.path.join(data_path, 'image')
data_lbl_path = os.path.join(data_path, 'label')
dest_img_path_train = os.path.join(dest_path_train, 'image'); mkdirfun(dest_img_path_train)
dest_lbl_path_train = os.path.join(dest_path_train, 'label'); mkdirfun(dest_lbl_path_train)
dest_img_path_valid = os.path.join(dest_path_valid, 'image'); mkdirfun(dest_img_path_valid)
dest_lbl_path_valid = os.path.join(dest_path_valid, 'label'); mkdirfun(dest_lbl_path_valid)

images = sorted(next(os.walk(data_img_path))[2])
subjects = []
for subject in images:
    if 'ED' in subject:
        subjects.append(subject.split('_')[0])

n_subjects = len(subjects)
random.shuffle(subjects)

n_train = int(n_subjects * 0.8)
for phase in ['ED','ES']:
    for subject in subjects[:n_train]:
        imagename1 = subject + '_' + phase + '_gt.nii.gz'
        imagename2 = subject + '_' + phase + '.nii.gz'
        os.symlink(data_img_path + '/' + imagename2, dest_img_path_train + '/' + imagename2)
        os.symlink(data_lbl_path + '/' + imagename1, dest_lbl_path_train + '/' + imagename2)

    for subject in subjects[n_train:]:
        imagename1 = subject + '_' + phase + '_gt.nii.gz'
        imagename2 = subject + '_' + phase + '.nii.gz'
        os.symlink(data_img_path + '/' + imagename2, dest_img_path_valid + '/' + imagename2)
        os.symlink(data_lbl_path + '/' + imagename1, dest_lbl_path_valid + '/' + imagename2)