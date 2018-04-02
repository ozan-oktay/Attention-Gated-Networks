# Author: Wenjia Bai
# Date: Feb 6th 2018
# Modified by: Ozan Oktay

# Perform intensity normalisation for LVSC 2009 dataset
import os
import numpy as np
import nibabel as nib


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


def learn_landmarks(image_path, savefilename, n_images=30):
    subjects = sorted(next(os.walk(image_path))[2])
    # choose the first n_images
    subjects = subjects[:n_images]
    image_data = []
    for subject in subjects:
        image_data.append(np.array(nib.load(os.path.join(image_path,subject)).get_data(), dtype=np.uint16).flatten())
    image_data = np.concatenate(image_data, axis=0)
    percentiles = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    target_landmarks = np.percentile(image_data, percentiles)
    np.save(savefilename, target_landmarks, allow_pickle=True, fix_imports=True)

    return target_landmarks

def normalise_image(imagepath, savepath):
    # Target histogram to match to
    image_path = '/vol/biomedic2/oo2113/dataset/UKBB_2964/sax/train/image'
    savefilename = '/vol/biomedic2/oo2113/projects/syntAI/tmp/target_lm_nyul.npy'
    if os.path.exists(savefilename):
        target_landmarks = np.load(savefilename, allow_pickle=True, fix_imports=True)
    else:
        target_landmarks = learn_landmarks(image_path, savefilename, n_images=50)

    # Perform intensity normalisation for ED and ES frames
    nim = nib.load(imagepath)
    image = nim.get_data()
    print(image.shape)
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
    nib.save(nim2, savepath)
