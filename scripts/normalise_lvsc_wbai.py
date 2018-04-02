# Author: Wenjia Bai
# Date: Feb 6th 2018
# Modified by: Ozan Oktay

# Perform intensity normalisation for LVSC 2009 dataset
import os
import numpy as np
import nibabel as nib


def get_roi(label, k=2, pad=50):
    X, Y, Z = label.shape
    roi = np.nonzero(label == k)
    x_min, y_min, _ = [x.min() for x in roi]
    x_max, y_max, _ = [x.max() for x in roi]
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(X, x_max + pad)
    y_max = min(Y, y_max + pad)
    return x_min, x_max, y_min, y_max


def nyul_normalisation(val, landmarks, target_landmarks):
    # Check which region the intensity value lies in
    n = len(landmarks)
    for i in range(n):
        if val < landmarks[i]:
            break

    # Extrapolate for the intensity below the first landmark
    if i == 0:
        i = 1

    # Landmark intensities
    # p: input; q: target
    p1 = landmarks[i - 1]
    p2 = landmarks[i]
    q1 = target_landmarks[i - 1]
    q2 = target_landmarks[i]

    # Linear rescaling
    val2 = (val - p1) / (p2 - p1) * (q2 - q1) + q1
    return val2


# Target histogram to match to
# TODO: an average histogram from a number of subjects
target_dir = '/vol/medic02/users/wbai/data/cardiac_atlas/UKBB_2964/data/1003105'
image_name = '{0}/sa.nii.gz'.format(target_dir)
image = nib.load(image_name).get_data()
label_name = '{0}/label_sa_ED.nii.gz'.format(target_dir)
label = nib.load(label_name).get_data()
percentiles = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
target_landmarks = np.percentile(image, percentiles)
print(target_landmarks)

# Data path
data_path = '/vol/biomedic2/oo2113/dataset/LVSC_2009/challenge_training'
dest_path = '/vol/biomedic2/oo2113/dataset/tmp'
data_list = sorted(os.listdir(data_path))

for data in data_list:
    print(data)
    data_dir = os.path.join(data_path, data)
    dest_dir = os.path.join(dest_path, data)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Perform intensity normalisation for ED and ES frames
    for fr in ['ED', 'ES']:
        image_name = '{0}/image_{1}.nii.gz'.format(data_dir, fr)
        nim = nib.load(image_name)
        image = nim.get_data()
        X, Y, Z = image.shape

        # Image histogram
        percentiles = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
        landmarks = np.percentile(image, percentiles)
        print(landmarks)

        # Perform Nyul's histogram matching
        # Piece-wise linear mapping
        for z in range(Z):
            for y in range(Y):
                for x in range(X):
                    val = image[x, y, z]
                    val2 = nyul_normalisation(val, landmarks, target_landmarks)
                    image[x, y, z] = val2

        nim2 = nib.Nifti1Image(image, nim.affine)
        nib.save(nim2, '{0}/image_{1}.nii.gz'.format(dest_dir, fr))



