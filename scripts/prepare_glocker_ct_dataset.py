import os
import subprocess
import numpy
import SimpleITK as sitk


def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def callmyfunction(mycmd):
    cmd = subprocess.Popen(mycmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
    stdoutput = cmd.communicate()[0].decode("utf-8") .strip('\n')
    print(stdoutput)
    return stdoutput


def exclude_images(input_list, query):
    new_list = []
    for name in input_list:
        if query not in name:
            new_list.append(name)
    return new_list


# Reverse axes
def reverse_image_zdir(img_path):
    flipAboutOrigin = False
    img = sitk.ReadImage(img_path)
    flipped_img = sitk.Flip(img, [False, True, True], flipAboutOrigin)
    sitk.WriteImage(flipped_img, img_path)


# Crop Image
def crop_ct_image(img_path, lbl_path, label_ids):
    tol = int(5)
    img = sitk.ReadImage(img_path)
    lbl = sitk.ReadImage(lbl_path)
    lbl_arr = sitk.GetArrayFromImage(lbl)
    [size_z, _, _] = lbl_arr.shape
    min_z = size_z -1
    max_z = 0
    for lbl_id in label_ids:
        [z, _, _] = numpy.where(lbl_arr == int(lbl_id))
        q_min_z = numpy.min(z) - tol
        q_max_z = numpy.max(z) + tol
        if q_min_z < min_z: min_z = q_min_z
        if q_max_z > max_z: max_z = q_max_z
    if min_z < 0: min_z = 0
    if max_z > size_z-1: max_z = size_z-1
    img_cropped = sitk.Crop(img, [0, 0, min_z], [0, 0, size_z-max_z])
    lbl_cropped = sitk.Crop(lbl, [0, 0, min_z], [0, 0, size_z-max_z])
    sitk.WriteImage(img_cropped, img_path)
    sitk.WriteImage(lbl_cropped, lbl_path)


# Target Splits
target_splits = ['train', 'validation']
split_ratio = 0.80

# Source Image Directories
source_dir = '/vol/biomedic2/bglocker/multi-organ/CT/Training'
source_img_dir = os.path.join(source_dir, 'img')
source_lbl_dir = os.path.join(source_dir, 'label')

# 1-> spleen
# 2-> liver
# 11 -> pancreas

# Class Labels
selected_class_ids = ['1', '2', '11']
new_class_ids = ['1', '3', '2']
selected_class_ids_str = ' '.join(selected_class_ids)
new_class_ids_str = ' '.join(new_class_ids)

# Split the dataset into training and validation sets
tmp_source_images = sorted(next(os.walk(source_img_dir))[2]); tmp_source_images = sorted(exclude_images(tmp_source_images, 'reoriented'))
tmp_source_labels = sorted(next(os.walk(source_lbl_dir))[2]); tmp_source_labels = sorted(exclude_images(tmp_source_labels, 'reoriented'))

source_images = {split: [] for split in target_splits}
source_labels = {split: [] for split in target_splits}
assert len(tmp_source_images) == len(tmp_source_labels)
for img, lbl in zip(tmp_source_images, tmp_source_labels):
    if numpy.random.uniform(low=0.0, high=1.0) <= split_ratio:
        source_images['train'].append(img)
        source_labels['train'].append(lbl)
    else:
        source_images['validation'].append(img)
        source_labels['validation'].append(lbl)

# Target Image Directories
target_dir = '/vol/biomedic2/oo2113/dataset/glocker_abdominal_ct'; mkdirfun(target_dir)
for split in target_splits:
    sub_target_dir = os.path.join(target_dir, split)
    target_img_dir = os.path.join(sub_target_dir, 'image'); mkdirfun(target_img_dir)
    target_lbl_dir = os.path.join(sub_target_dir, 'label'); mkdirfun(target_lbl_dir)

    for source_img, source_lbl in zip(source_images[split], source_labels[split]):
        print(source_img)
        print(source_lbl)

        source_img_fullpath = os.path.join(source_img_dir, source_img)
        source_lbl_fullpath = os.path.join(source_lbl_dir, source_lbl)
        target_img_fullpath = os.path.join(target_img_dir, source_img)
        target_lbl_fullpath = os.path.join(target_lbl_dir, source_img)

        # Resample the images to 2mm isotropic resolution
        callmyfunction('blur {0} {1} 0.75 -3D -short'.format(source_img_fullpath, target_img_fullpath))
        callmyfunction('resample {0} {1} -size 2.00 2.00 2.00 -linear'.format(target_img_fullpath, target_img_fullpath))
        callmyfunction('resample {0} {1} -size 2.00 2.00 2.00 -nn'.format(source_lbl_fullpath, target_lbl_fullpath))

        # Select only the spleen pancreas and kidneys
        callmyfunction('changeLabelId.py --inputname {0} --outputname {0} -o 3 -n 2'.format(target_lbl_fullpath))
        crop_ct_image(target_img_fullpath, target_lbl_fullpath, label_ids=selected_class_ids)
        callmyfunction('selectLabelId.py --inputname {0} --outputname {0} -l {1}'.format(target_lbl_fullpath, selected_class_ids_str))
        callmyfunction('changeLabelId.py --inputname {0} --outputname {0} -o {1} -n {2}'.format(target_lbl_fullpath, selected_class_ids_str, new_class_ids_str))

        # Flip the axis
        reverse_image_zdir(target_img_fullpath)
        reverse_image_zdir(target_lbl_fullpath)
        callmyfunction('headertool {0} {0} -reset'.format(target_img_fullpath))
        callmyfunction('headertool {0} {0} -reset'.format(target_lbl_fullpath))
