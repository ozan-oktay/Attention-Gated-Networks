import numpy as np
import math
import torch.nn as nn
from .utils import unetConv2, unetUp, conv2DBatchNormRelu, conv2DBatchNorm
import torch.nn.functional as F
from models.networks_other import init_weights

class sononet(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, in_channels=3, is_batchnorm=True, n_convs=None):
        super(sononet, self).__init__()
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.n_classes= n_classes

        filters = [64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        if n_convs is None:
            n_convs = [2,2,3,3,3]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, n=n_convs[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, n=n_convs[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, n=n_convs[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, n=n_convs[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[3], self.is_batchnorm, n=n_convs[4])

        # adaptation layer
        self.conv5_p = conv2DBatchNormRelu(filters[3], filters[2], 1, 1, 0)
        self.conv6_p = conv2DBatchNorm(filters[2], self.n_classes, 1, 1, 0)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')


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

        conv5_p  = self.conv5_p(conv5)
        conv6_p  = self.conv6_p(conv5_p)

        batch_size = inputs.shape[0]
        pooled     = F.adaptive_avg_pool2d(conv6_p, (1, 1)).view(batch_size, -1)

        return pooled


    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


def sononet2(feature_scale=4, n_classes=21, in_channels=3, is_batchnorm=True):
    return sononet(feature_scale, n_classes, in_channels, is_batchnorm, n_convs=[3,3,3,2,2])

