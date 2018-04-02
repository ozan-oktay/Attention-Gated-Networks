import math
import torch.nn as nn
from .utils import UnetConv3, UnetUp3
import torch.nn.functional as F
from models.layers.nonlocal_layer import NONLocalBlock3D
from models.networks_other import init_weights

class unet_nonlocal_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True,
                 nonlocal_mode='embedded_gaussian', nonlocal_sf=4):
        super(unet_nonlocal_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.nonlocal2 = NONLocalBlock3D(in_channels=filters[1], inter_channels=filters[1] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.nonlocal3 = NONLocalBlock3D(in_channels=filters[2], inter_channels=filters[2] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 1))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = UnetUp3(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = UnetUp3(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = UnetUp3(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = UnetUp3(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        nl2 = self.nonlocal2(conv2)
        maxpool2 = self.maxpool2(nl2)

        conv3 = self.conv3(maxpool2)
        nl3 = self.nonlocal3(conv3)
        maxpool3 = self.maxpool3(nl3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(nl3,   up4)
        up2 = self.up_concat2(nl2,   up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p













