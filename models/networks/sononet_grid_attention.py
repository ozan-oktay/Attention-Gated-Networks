import numpy as np
import math
import torch.nn as nn
from .utils import unetConv2, unetUp, conv2DBatchNormRelu, conv2DBatchNorm
import torch
import torch.nn.functional as F
from models.layers.grid_attention_layer import GridAttentionBlock2D_TORR as AttentionBlock2D
from models.networks_other import init_weights

class sononet_grid_attention(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, in_channels=3, is_batchnorm=True, n_convs=None,
                 nonlocal_mode='concatenation', aggregation_mode='concat'):
        super(sononet_grid_attention, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes= n_classes
        self.aggregation_mode = aggregation_mode
        self.deep_supervised = True

        if n_convs is None:
            n_convs = [3, 3, 3, 2, 2]

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        ####################
        # Feature Extraction
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, n=n_convs[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, n=n_convs[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, n=n_convs[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, n=n_convs[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[3], self.is_batchnorm, n=n_convs[4])

        ################
        # Attention Maps
        self.compatibility_score1 = AttentionBlock2D(in_channels=filters[2], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        self.compatibility_score2 = AttentionBlock2D(in_channels=filters[3], gating_channels=filters[3],
                                                     inter_channels=filters[3], sub_sample_factor=(1,1),
                                                     mode=nonlocal_mode, use_W=False, use_phi=True,
                                                     use_theta=True, use_psi=True, nonlinearity1='relu')

        #########################
        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[2], filters[3]]

        if aggregation_mode == 'concat':
            self.classifier = nn.Linear(filters[2]+filters[3]+filters[3], n_classes)
            self.aggregate = self.aggreagation_concat

        else:
            self.classifier1 = nn.Linear(filters[2], n_classes)
            self.classifier2 = nn.Linear(filters[3], n_classes)
            self.classifier3 = nn.Linear(filters[3], n_classes)
            self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

            if aggregation_mode == 'mean':
                self.aggregate = self.aggregation_sep

            elif aggregation_mode == 'deep_sup':
                self.classifier = nn.Linear(filters[2] + filters[3] + filters[3], n_classes)
                self.aggregate = self.aggregation_ds

            elif aggregation_mode == 'ft':
                self.classifier = nn.Linear(n_classes*3, n_classes)
                self.aggregate = self.aggregation_ft
            else:
                raise NotImplementedError

        ####################
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


    def aggregation_sep(self, *attended_maps):
        return [ clf(att) for clf, att in zip(self.classifiers, attended_maps) ]

    def aggregation_ft(self, *attended_maps):
        preds =  self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep =  self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))


    def forward(self, inputs):
        # Feature Extraction
        conv1    = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2    = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3    = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4    = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        conv5    = self.conv5(maxpool4)

        batch_size = inputs.shape[0]
        pooled     = F.adaptive_avg_pool2d(conv5, (1, 1)).view(batch_size, -1)

        # Attention Mechanism
        g_conv1, att1 = self.compatibility_score1(conv3, conv5)
        g_conv2, att2 = self.compatibility_score2(conv4, conv5)

        # flatten to get single feature vector
        fsizes = self.attention_filter_sizes
        g1 = torch.sum(g_conv1.view(batch_size, fsizes[0], -1), dim=-1)
        g2 = torch.sum(g_conv2.view(batch_size, fsizes[1], -1), dim=-1)

        return self.aggregate(g1, g2, pooled)


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
