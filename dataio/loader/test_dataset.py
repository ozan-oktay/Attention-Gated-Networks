import torch.utils.data as data
import numpy as np
import os

from os import listdir
from os.path import join
from .utils import load_nifti_img, check_exceptions, is_image_file


class TestDataset(data.Dataset):
    def __init__(self, root_dir, transform):
        super(TestDataset, self).__init__()
        image_dir = join(root_dir, 'image')
        self.image_filenames = sorted([join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)])

        # Add the corresponding ground-truth images if they exist
        self.label_filenames = []
        label_dir = join(root_dir, 'label')
        if os.path.isdir(label_dir):
            self.label_filenames = sorted([join(label_dir, x) for x in listdir(label_dir) if is_image_file(x)])
            assert len(self.label_filenames) == len(self.image_filenames)

        # data pre-processing
        self.transform = transform

        # report the number of images in the dataset
        print('Number of test images: {0} NIFTIs'.format(self.__len__()))

    def __getitem__(self, index):

        # load the NIFTI images
        input, input_meta = load_nifti_img(self.image_filenames[index], dtype=np.int16)

        # load the label image if it exists
        if self.label_filenames:
            label, _ = load_nifti_img(self.label_filenames[index], dtype=np.int16)
            check_exceptions(input, label)
        else:
            label = []
            check_exceptions(input)

        # Pre-process the input 3D Nifti image
        input = self.transform(input)

        return input, input_meta, label

    def __len__(self):
        return len(self.image_filenames)