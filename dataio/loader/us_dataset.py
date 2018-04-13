import torch
import torch.utils.data as data
import h5py
import numpy as np
import datetime

from os import listdir
from os.path import join
#from .utils import check_exceptions


class UltraSoundDataset(data.Dataset):
    def __init__(self, root_path, split, transform=None, preload_data=False):
        super(UltraSoundDataset, self).__init__()

        f = h5py.File(root_path)

        self.images = f['x_'+split]

        if preload_data:
            self.images = np.array(self.images[:])

        self.labels = np.array(f['p_'+split][:], dtype=np.int64)#[:1000]
        self.label_names = [x.decode('utf-8') for x in f['label_names'][:].tolist()]
        #print(self.label_names)
        #print(np.unique(self.labels[:]))
        # construct weight for entry
        self.n_class = len(self.label_names)
        class_weight = np.zeros(self.n_class)
        for lab in range(self.n_class):
            class_weight[lab] = np.sum(self.labels[:] == lab)

        class_weight = 1 / class_weight
    
        self.weight = np.zeros(len(self.labels))
        for i in range(len(self.labels)):
            self.weight[i] = class_weight[self.labels[i]]

        #print(class_weight)
        assert len(self.images) == len(self.labels)

        # data augmentation
        self.transform = transform

        # report the number of images in the dataset
        print('Number of {0} images: {1} NIFTIs'.format(split, self.__len__()))

    def __getitem__(self, index):
        # update the seed to avoid workers sample the same augmentation parameters
        np.random.seed(datetime.datetime.now().second + datetime.datetime.now().microsecond)

        # load the nifti images
        input  = self.images[index][0]
        target = self.labels[index]

        #input = input.transpose((1,2,0))

        # handle exceptions
        #check_exceptions(input, target)
        if self.transform:
            input = self.transform(input)

        #print(input.shape, torch.from_numpy(np.array([target])))
        #print("target",np.int64(target))
        return input, int(target)

    def __len__(self):
        return len(self.images)


# if __name__ == '__main__':
#     dataset = UltraSoundDataset("/vol/bitbucket/js3611/data_ultrasound/preproc_combined_inp_224x288.hdf5",'test')

#     from torch.utils.data import DataLoader, sampler
#     ds = DataLoader(dataset=dataset, num_workers=1, batch_size=2)
