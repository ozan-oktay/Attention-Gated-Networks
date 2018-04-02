from torch.utils.data import DataLoader

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from models import get_model

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math, numpy, os
from scipy.misc import imresize
from skimage.transform import resize
from dataio.loader.utils import write_nifti_img
from torch.nn import functional as F

def plotNNFilter(units, figure_id, interp='bilinear', colormap=cm.jet, colormap_lim=None):
    plt.ion()
    filters = units.shape[2]
    n_columns = round(math.sqrt(filters))
    n_rows = math.ceil(filters / n_columns) + 1
    fig = plt.figure(figure_id, figsize=(n_rows*3,n_columns*3))
    fig.clf()

    for i in range(filters):
        ax1 = plt.subplot(n_rows, n_columns, i+1)
        plt.imshow(units[:,:,i].T, interpolation=interp, cmap=colormap)
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        plt.colorbar()
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

# Load options
json_opts = json_file_to_pyobj('/vol/biomedic2/oo2113/projects/syntAI/ukbb_pytorch/configs_final/debug_ct.json')

# Setup the NN Model
model = get_model(json_opts.model)

# Setup Dataset and Augmentation
dataset_class = get_dataset('test_sax')
dataset_path = get_dataset_path('test_sax', json_opts.data_path)
dataset_transform = get_dataset_transformation('test_sax', json_opts.augmentation)

# Setup Data Loader
dataset = dataset_class(dataset_path, transform=dataset_transform['test'])
data_loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=False)

# test
for iteration, (input_arr, input_meta, _) in enumerate(data_loader, 1):
    model.set_input(input_arr)
    layer_name = 'attentionblock1'
    inp_fmap, out_fmap = model.get_feature_maps(layer_name=layer_name, upscale=False)

    # Display the input image and Down_sample the input image
    orig_input_img = model.input.permute(2, 3, 4, 1, 0).cpu().numpy()
    upsampled_attention   = F.upsample(out_fmap[1], size=input_arr.size()[2:], mode='trilinear').data.squeeze().permute(1,2,3,0).cpu().numpy()
    upsampled_fmap_before = F.upsample(inp_fmap[0], size=input_arr.size()[2:], mode='trilinear').data.squeeze().permute(1,2,3,0).cpu().numpy()
    upsampled_fmap_after  = F.upsample(out_fmap[2], size=input_arr.size()[2:], mode='trilinear').data.squeeze().permute(1,2,3,0).cpu().numpy()

    # Define the directories
    save_directory = os.path.join('/vol/bitbucket/oo2113/tmp/feature_maps', layer_name)
    basename = input_meta['name'][0].split('.')[0]

    # Write the attentions to a nifti image
    input_meta['name'][0] = basename + '_img.nii.gz'
    write_nifti_img(orig_input_img, input_meta, savedir=save_directory)

    input_meta['name'][0] = basename + '_att.nii.gz'
    write_nifti_img(upsampled_attention, input_meta, savedir=save_directory)

    input_meta['name'][0] = basename + '_fmap_before.nii.gz'
    write_nifti_img(upsampled_fmap_before, input_meta, savedir=save_directory)

    input_meta['name'][0] = basename + '_fmap_after.nii.gz'
    write_nifti_img(upsampled_fmap_after, input_meta, savedir=save_directory)

model.destructor()
#if iteration == 1: break