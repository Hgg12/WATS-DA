# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch

def pixel_wise_corr(z, x):
    '''
    z is kernel ([32, 96, 8, 8])
    x is search ([32, 96, 16, 16])

    z -> (32, 64, 96)
    x -> (32, 96, 256)
    '''
    b, c, h, w = x.size()
    z_mat = z.contiguous().view((b, c, -1)).transpose(1, 2)  # (b,64,c)
    x_mat = x.contiguous().view((b, c, -1))  # (b,c,256)
    return torch.matmul(z_mat, x_mat).view((b, -1, h, w))


class SE(nn.Module):

    def __init__(self, channels=49, reduction=1):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)  # nn.silu()
        return module_input * x


class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        if x.size(3) < 20:
            l = 4
            r = l + 7
            x = x[:, :, l:r, l:r]
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out

class EFM(nn.Module):
    def __init__(self, num_kernel=49, adj_channel=256):
        super().__init__()

        # pw-corr
        self.pw_corr = pixel_wise_corr
        self.ca = SE(num_kernel)
        # self.ca = eca(num_kernel)

        # SCF
        # self.conv55 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=5, stride=1, padding=2,
        #                         groups=num_kernel)
        # self.bn55 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)


        self.conv33 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=3, stride=1, padding=1,
                                groups=num_kernel)
        self.bn33 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.conv11 = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel, kernel_size=1, stride=1, padding=0,
                                groups=num_kernel)
        self.bn11 = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        # IAB
        self.conv_up = nn.Conv2d(in_channels=num_kernel, out_channels=num_kernel * 2, kernel_size=1, stride=1,
                                 padding=0)
        self.bn_up = nn.BatchNorm2d(num_kernel * 2, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)
        self.act = nn.GELU()

        self.conv_down = nn.Conv2d(in_channels=num_kernel * 2, out_channels=num_kernel, kernel_size=1, stride=1,
                                   padding=0)
        self.bn_down = nn.BatchNorm2d(num_kernel, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True)

        self.adjust = nn.Conv2d(num_kernel, adj_channel, 1)

    def forward(self, z, x):
        corr = self.pw_corr(z, x)
        corr = self.ca(corr)

        # scf + skip-connection
        corr = corr + self.bn11(self.conv11(corr)) + self.bn33(self.conv33(corr)) 

        # iab + skip-connection
        corr = corr + self.bn_down(self.conv_down(self.act(self.bn_up(self.conv_up(corr)))))

        corr = self.adjust(corr)

        # corr = torch.cat((corr, x), dim=1)
        corr = corr + x

        return corr