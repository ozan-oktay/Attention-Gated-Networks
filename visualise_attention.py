from torch.utils.data import DataLoader

from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser
from models import get_model
import os, time

# import matplotlib
# matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import math, numpy
import numpy as np 
from scipy.misc import imresize
from skimage.transform import resize

def plotNNFilter(units, figure_id, interp='bilinear', colormap=cm.jet, colormap_lim=None, title=''):
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
    plt.suptitle(title)

def plotNNFilterOverlay(input_im, units, figure_id, interp='bilinear',
                        colormap=cm.jet, colormap_lim=None, title='', alpha=0.8):
    plt.ion()
    filters = units.shape[2]
    fig = plt.figure(figure_id, figsize=(5,5))
    fig.clf()

    for i in range(filters):
        plt.imshow(input_im[:,:,0], interpolation=interp, cmap='gray')
        plt.imshow(units[:,:,i], interpolation=interp, cmap=colormap, alpha=alpha)
        plt.axis('off')
        plt.colorbar()
        plt.title(title, fontsize='small')
        if colormap_lim:
            plt.clim(colormap_lim[0],colormap_lim[1])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()

    # plt.savefig('{}/{}.png'.format(dir_name,time.time()))




## Load options
PAUSE = .01
#config_name = 'config_sononet_attention_fs8_v6.json'
#config_name = 'config_sononet_attention_fs8_v8.json'
#config_name = 'config_sononet_attention_fs8_v9.json'
#config_name = 'config_sononet_attention_fs8_v10.json'
#config_name = 'config_sononet_attention_fs8_v11.json'
#config_name = 'config_sononet_attention_fs8_v13.json'
#config_name = 'config_sononet_attention_fs8_v14.json'
#config_name = 'config_sononet_attention_fs8_v15.json'
#config_name = 'config_sononet_attention_fs8_v16.json'
#config_name = 'config_sononet_grid_attention_fs8_v1.json'
config_name = 'config_sononet_grid_attention_fs8_deepsup_v1.json'
config_name = 'config_sononet_grid_attention_fs8_deepsup_v2.json'
config_name = 'config_sononet_grid_attention_fs8_deepsup_v3.json'
config_name = 'config_sononet_grid_attention_fs8_deepsup_v4.json'

# config_name = 'config_sononet_grid_att_fs8_avg.json'
config_name = 'config_sononet_grid_att_fs8_avg_v2.json'
# config_name = 'config_sononet_grid_att_fs8_avg_v3.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v4.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v5.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v5.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v6.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v7.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v8.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v9.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v10.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v11.json'
#config_name = 'config_sononet_grid_att_fs8_avg_v12.json'

config_name = 'config_sononet_grid_att_fs8_avg_v12_scratch.json'
config_name = 'config_sononet_grid_att_fs4_avg_v12.json'

#config_name = 'config_sononet_grid_attention_fs8_v3.json'

json_opts = json_file_to_pyobj('/vol/bitbucket/js3611/projects/transfer_learning/ultrasound/configs_2/{}'.format(config_name))
train_opts = json_opts.training

dir_name = os.path.join('visualisation_debug', config_name)
if not os.path.isdir(dir_name):
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name,'pos'))
    os.makedirs(os.path.join(dir_name,'neg'))

# Setup the NN Model
model = get_model(json_opts.model)
if hasattr(model.net, 'classification_mode'):
    model.net.classification_mode = 'attention'
if hasattr(model.net, 'deep_supervised'):
    model.net.deep_supervised = False 

# Setup Dataset and Augmentation
dataset_class = get_dataset(train_opts.arch_type)
dataset_path = get_dataset_path(train_opts.arch_type, json_opts.data_path)
dataset_transform = get_dataset_transformation(train_opts.arch_type, opts=json_opts.augmentation)

# Setup Data Loader
dataset = dataset_class(dataset_path, split='train', transform=dataset_transform['valid'])
data_loader = DataLoader(dataset=dataset, num_workers=1, batch_size=1, shuffle=True)

# test
for iteration, data in enumerate(data_loader, 1):
    model.set_input(data[0], data[1])

    cls = dataset.label_names[int(data[1])]

    model.validate()
    pred_class = model.pred[1]
    pred_cls = dataset.label_names[int(pred_class)]

    #########################################################
    # Display the input image and Down_sample the input image
    input_img = model.input[0,0].cpu().numpy()
    #input_img = numpy.expand_dims(imresize(input_img, (fmap_size[0], fmap_size[1]), interp='bilinear'), axis=2)
    input_img = numpy.expand_dims(input_img, axis=2)

    # plotNNFilter(input_img, figure_id=0, colormap="gray")
    plotNNFilterOverlay(input_img, numpy.zeros_like(input_img), figure_id=0, interp='bilinear',
                        colormap=cm.jet, title='[GT:{}|P:{}]'.format(cls, pred_cls),alpha=0)

    chance = np.random.random() < 0.01 if cls == "BACKGROUND" else 1
    if cls != pred_cls:
        plt.savefig('{}/neg/{:03d}.png'.format(dir_name,iteration))
    elif cls == pred_cls and chance:
        plt.savefig('{}/pos/{:03d}.png'.format(dir_name,iteration))
    #########################################################
    # Compatibility Scores overlay with input
    attentions = []
    for i in [1,2]:
        fmap = model.get_feature_maps('compatibility_score%d'%i, upscale=False)
        if not fmap:
            continue

        # Output of the attention block
        fmap_0 = fmap[0].squeeze().permute(1,2,0).cpu().numpy()
        fmap_size = fmap_0.shape

        # Attention coefficient (b x c x w x h x s)
        attention = fmap[1].squeeze().cpu().numpy()
        attention = attention[:, :]
        #attention = numpy.expand_dims(resize(attention, (fmap_size[0], fmap_size[1]), mode='constant', preserve_range=True), axis=2)
        attention = numpy.expand_dims(resize(attention, (input_img.shape[0], input_img.shape[1]), mode='constant', preserve_range=True), axis=2)

        # this one is useless
        #plotNNFilter(fmap_0, figure_id=i+3, interp='bilinear', colormap=cm.jet, title='compat. feature %d' %i)

        plotNNFilterOverlay(input_img, attention, figure_id=i, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. {}'.format(cls,pred_cls,i), alpha=0.5)
        attentions.append(attention)

    #plotNNFilterOverlay(input_img, attentions[0], figure_id=4, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. (all)'.format(cls, pred_cls), alpha=0.5)
    plotNNFilterOverlay(input_img, numpy.mean(attentions,0), figure_id=4, interp='bilinear', colormap=cm.jet, title='[GT:{}|P:{}] compat. (all)'.format(cls, pred_cls), alpha=0.5)

    if cls != pred_cls:
        plt.savefig('{}/neg/{:03d}_hm.png'.format(dir_name,iteration))
    elif cls == pred_cls and chance:
        plt.savefig('{}/pos/{:03d}_hm.png'.format(dir_name,iteration))
    # Linear embedding g(x)
    # (b, c, h, w)
    #gx = fmap[2].squeeze().permute(1,2,0).cpu().numpy()
    #plotNNFilter(gx, figure_id=3, interp='nearest', colormap=cm.jet)

    plt.show()
    plt.pause(PAUSE)

model.destructor()
#if iteration == 1: break
