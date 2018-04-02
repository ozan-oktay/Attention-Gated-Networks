import os
import subprocess
import numpy

def mkdirfun(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def callmyfunction(mycmd):
    cmd = subprocess.Popen(mycmd, shell=True, executable="/bin/bash", stdout=subprocess.PIPE)
    stdoutput = cmd.communicate()[0].decode("utf-8") .strip('\n')
    print(stdoutput)
    return stdoutput

# Target Splits
target_splits = ['train', 'validation']
split_ratio = 0.80

# Source Image Directories
source_dir = '/vol/bitbucket/oo2113/tmp/abdominal_ct'
source_img_dir = os.path.join(source_dir, 'imgs')
source_lbl_dir = os.path.join(source_dir, 'lbls')

# Class Ids
selected_class_ids = ['4', '7', '8']
new_class_ids = [str(ii) for ii in range(1,len(selected_class_ids)+1)]
selected_class_ids = ' '.join(selected_class_ids)
new_class_ids = ' '.join(new_class_ids)

# Split the dataset into training and validation sets
tmp_source_images = sorted(next(os.walk(source_img_dir))[2])
tmp_source_labels = sorted(next(os.walk(source_lbl_dir))[2])
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
target_dir = '/vol/biomedic2/oo2113/dataset/ken_abdominal_ct_15mm'; mkdirfun(target_dir)
for split in target_splits:
    sub_target_dir = os.path.join(target_dir, split)
    target_img_dir = os.path.join(sub_target_dir, 'image'); mkdirfun(target_img_dir)
    target_lbl_dir = os.path.join(sub_target_dir, 'label'); mkdirfun(target_lbl_dir)

    for source_img, source_lbl in zip(source_images[split], source_labels[split]):
        assert source_img == source_lbl
        source_img_fullpath = os.path.join(source_img_dir, source_img)
        source_lbl_fullpath = os.path.join(source_lbl_dir, source_lbl)
        target_img_fullpath = os.path.join(target_img_dir, source_img)
        target_lbl_fullpath = os.path.join(target_lbl_dir, source_lbl)

        # Resample the images to 2mm isotropic resolution
        callmyfunction('blur {0} {1} 0.50 -3D -short'.format(source_img_fullpath, target_img_fullpath))
        callmyfunction('resample {0} {1} -size 1.50 1.50 1.50 -linear'.format(target_img_fullpath, target_img_fullpath))
        callmyfunction('resample {0} {1} -size 1.50 1.50 1.50 -nn'.format(source_lbl_fullpath, target_lbl_fullpath))

        # Select only the spleen pancreas and kidneys
        callmyfunction('selectLabelId.py --inputname {0} --outputname {0} -l {1}'.format(target_lbl_fullpath, selected_class_ids))
        callmyfunction('changeLabelId.py --inputname {0} --outputname {0} -o {1} -n {2}'.format(target_lbl_fullpath, selected_class_ids, new_class_ids))