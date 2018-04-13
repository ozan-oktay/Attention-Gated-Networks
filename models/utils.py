'''
Misc Utility functions
'''

import os
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from utils.metrics import segmentation_scores, dice_score_list
from sklearn import metrics
from .layers.loss import *

def get_optimizer(option, params):
    opt_alg = 'sgd' if not hasattr(option, 'optim') else option.optim
    if opt_alg == 'sgd':
        optimizer = optim.SGD(params,
                              lr=option.lr_rate,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=option.l2_reg_weight)

    if opt_alg == 'adam':
        optimizer = optim.Adam(params,
                               lr=option.lr_rate,
                               betas=(0.9, 0.999),
                               weight_decay=option.l2_reg_weight)

    return optimizer


def get_criterion(opts):
    if opts.criterion == 'cross_entropy':
        if opts.type == 'seg':
            criterion = cross_entropy_2D if opts.tensor_dim == '2D' else cross_entropy_3D
        elif 'classifier' in opts.type:
            criterion = CrossEntropyLoss()
    elif opts.criterion == 'dice_loss':
        criterion = SoftDiceLoss(opts.output_nc)
    elif opts.criterion == 'dice_loss_pancreas_only':
        criterion = CustomSoftDiceLoss(opts.output_nc, class_ids=[0, 2])

    return criterion

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1, max_iter=30000, power=0.9,):
    """Polynomial decay of learning rate
        :param init_lr is base learning rate
        :param iter is a current iteration
        :param lr_decay_iter how frequently decay occurs, default is 1
        :param max_iter is number of maximum iterations
        :param power is a polymomial power

    """
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer

    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*(1 - iter/max_iter)**power


def adjust_learning_rate(optimizer, init_lr, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def segmentation_stats(pred_seg, target):
    n_classes = pred_seg.size(1)
    pred_lbls = pred_seg.data.max(1)[1].cpu().numpy()
    gt = np.squeeze(target.data.cpu().numpy(), axis=1)
    gts, preds = [], []
    for gt_, pred_ in zip(gt, pred_lbls):
        gts.append(gt_)
        preds.append(pred_)

    iou = segmentation_scores(gts, preds, n_class=n_classes)
    dice = dice_score_list(gts, preds, n_class=n_classes)

    return iou, dice


def classification_scores(gts, preds, labels):
    accuracy        = metrics.accuracy_score(gts,  preds)
    class_accuracies = []
    for lab in labels: # TODO Fix
        class_accuracies.append(metrics.accuracy_score(gts[gts == lab], preds[gts == lab]))
    class_accuracies = np.array(class_accuracies)

    f1_micro        = metrics.f1_score(gts,        preds, average='micro')
    precision_micro = metrics.precision_score(gts, preds, average='micro')
    recall_micro    = metrics.recall_score(gts,    preds, average='micro')
    f1_macro        = metrics.f1_score(gts,        preds, average='macro')
    precision_macro = metrics.precision_score(gts, preds, average='macro')
    recall_macro    = metrics.recall_score(gts,    preds, average='macro')

    # class wise score
    f1s        = metrics.f1_score(gts,        preds, average=None)
    precisions = metrics.precision_score(gts, preds, average=None)
    recalls    = metrics.recall_score(gts,    preds, average=None)

    confusion = metrics.confusion_matrix(gts,preds, labels=labels)

    #TODO confusion matrix, recall, precision
    return accuracy, f1_micro, precision_micro, recall_micro, f1_macro, precision_macro, recall_macro, confusion, class_accuracies, f1s, precisions, recalls


def classification_stats(pred_seg, target, labels):
    return classification_scores(target, pred_seg, labels)
