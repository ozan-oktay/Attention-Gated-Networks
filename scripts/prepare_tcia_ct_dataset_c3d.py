import os
import subprocess
import numpy
import shutil
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
def reverse_image_direction(img_path, bool_dir=(False, False, False)):
    flipAboutOrigin = False
    img = sitk.ReadImage(img_path)
    flipped_img = sitk.Flip(img, bool_dir, flipAboutOrigin)
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

# Return Subdirs
def return_subdirs(path):
    output_list = sorted([os.path.join(path,name) for name in os.listdir(path) if os.path.isdir(os.path.join(path,name))])
    return output_list

# Find all files with extension
def find_files_ext(directory, extension):
    output_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                output_list.append(os.path.join(root,file))
    return sorted(output_list), root

def correct_origin(imgpath, lblpath, offset_flip):
    offset_sign = -1.0 if offset_flip else 1.0
    int_img = sitk.ReadImage(imgpath)
    lbl_img = sitk.ReadImage(lblpath)
    int_size = int_img.GetSize()
    lbl_size = lbl_img.GetSize()
    lbl_origin = lbl_img.GetOrigin()
    lbl_spacing = lbl_img.GetSpacing()
    z_origin_offset = (float(lbl_size[2])/2.0 - float(int_size[2])/2.0) * float(lbl_spacing[2]) * offset_sign

    lbl_img.SetOrigin((lbl_origin[0], lbl_origin[1], lbl_origin[2]+z_origin_offset))

    sitk.WriteImage(lbl_img, lblpath)

# Source Image Directories
dcm_dirs = '/vol/bitbucket/oo2113/TCIA/Pancreas-CT/'
lbl_dir  = '/vol/bitbucket/oo2113/TCIA/TCIA_pancreas_labels-02-05-2017'
subject_dirs = return_subdirs(dcm_dirs)
target_img_dir = '/vol/bitbucket/oo2113/TCIA/nifti_imgs'; mkdirfun(target_img_dir)
target_lbl_dir = '/vol/bitbucket/oo2113/TCIA/nifti_lbls'; mkdirfun(target_lbl_dir)

# Loop over the subjects and convert niftis
for subject_dir in subject_dirs:

    # Reconstruct the nifti images from dcms
    subject_id = int(subject_dir.split('_')[-1])
    target_nifti_name = os.path.join(target_img_dir, 'image{0:04d}'.format(subject_id) + '.nii.gz')
    source_label_name = os.path.join(lbl_dir, 'label{0:04d}'.format(subject_id) + '.nii.gz')
    target_label_name = os.path.join(target_lbl_dir, 'image{0:04d}'.format(subject_id) + '.nii.gz')
    dcm_files, dcm_dir= find_files_ext(subject_dir, '.dcm')
    std_out = callmyfunction('c3d -dicom-series-list {0}'.format(dcm_dir))
    series_id = std_out.split('Pancreas')[-1]
    
    # Create the nifti files
    callmyfunction('c3d -dicom-series-read {0} {1} -type short -o {2}'.format(dcm_dir, series_id, target_nifti_name))
    shutil.copy(source_label_name, target_label_name)

    # Resample the intensity image to isotropic 2mm resolution
    callmyfunction('blur {0} {0} 0.75 -3D -short'.format(target_nifti_name))
    callmyfunction('resample {0} {0} -size 2.00 2.00 2.00 -linear'.format(target_nifti_name))
    callmyfunction('resample {0} {0} -size 2.00 2.00 2.00 -nn'.format(target_label_name))

    # Flip the orientation
    reverse_image_direction(target_nifti_name, bool_dir=[False, True, False])
    reverse_image_direction(target_label_name, bool_dir=[False, True, False])
    callmyfunction('headertool {0} {0} -reset'.format(target_nifti_name))
    callmyfunction('headertool {0} {0} -reset'.format(target_label_name))

    # Change the pancreas label to class_id=2
    callmyfunction('changeLabelId.py --inputname {0} --outputname {0} -o 1 -n 2'.format(target_label_name))
