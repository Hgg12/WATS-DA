# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
from pysot.models.neck.tran import Transformer
import torch
import math

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

# # --------------------------------------------------------- #
# #（2）ECANet 通道注意力机制
# # 使用1D卷积代替SE注意力机制中的全连接层
# # --------------------------------------------------------- #

# # 定义ECANet的类
# class eca(nn.Module):
#     # 初始化, in_channel代表特征图的输入通道数, b和gama代表公式中的两个系数
#     def __init__(self, in_channel, b=1, gama=2):
#         # 继承父类初始化
#         super(eca, self).__init__()
        
#         # 根据输入通道数自适应调整卷积核大小
#         kernel_size = int(abs((math.log(in_channel, 2)+b)/gama))
#         # 如果卷积核大小是奇数，就使用它
#         if kernel_size % 2:
#             kernel_size = kernel_size
#         # 如果卷积核大小是偶数，就把它变成奇数
#         else:
#             kernel_size = kernel_size
        
#         # 卷积时，为例保证卷积前后的size不变，需要0填充的数量
#         padding = kernel_size // 2
        
#         # 全局平均池化，输出的特征图的宽高=1
#         self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
#         # 1D卷积，输入和输出通道数都=1，卷积核大小是自适应的
#         self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size,
#                               bias=False, padding=padding)
#         # sigmoid激活函数，权值归一化
#         self.sigmoid = nn.Sigmoid()
    
#     # 前向传播
#     def forward(self, inputs):
#         # 获得输入图像的shape
#         b, c, h, w = inputs.shape
        
#         # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
#         x = self.avg_pool(inputs)
#         # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
#         x = x.view([b,1,c])
#         # 1D卷积 [b,1,c]==>[b,1,c]
#         x = self.conv(x)
#         # 权值归一化
#         x = self.sigmoid(x)
#         # 维度调整 [b,1,c]==>[b,c,1,1]
#         x = x.view([b,c,1,1])
        
#         # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
#         outputs = x * inputs
#         return outputs


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
                out.append(adj_layer(features[i]).contiguous())
            return out

class Adjust_Transformer(nn.Module):
    def __init__(self, channels=256):
        super(Adjust_Transformer, self).__init__()

        self.row_embed = nn.Embedding(50, channels//2)
        self.col_embed = nn.Embedding(50, channels//2)
        self.reset_parameters()

        self.transformer = Transformer(channels, nhead = 8, num_encoder_layers = 1, num_decoder_layers = 0)

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x_f):
        # adjust search features
        h, w = x_f.shape[-2:]
        i = torch.arange(w).cuda()
        j = torch.arange(h).cuda()
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
            ], dim= -1).permute(2, 0, 1).unsqueeze(0).repeat(x_f.shape[0], 1, 1, 1)
        b, c, w, h = x_f.size()
        x_f = self.transformer((pos+x_f).view(b, c, -1).permute(2, 0, 1),\
                                (pos+x_f).view(b, c, -1).permute(2, 0, 1),\
                                    (pos+x_f).view(b, c, -1).permute(2, 0, 1))
        x_f = x_f.permute(1, 2, 0).view(b, c, w, h)

        return x_f


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
# if __name__ == '__main__':
#     x = torch.randn(1, 256, 15, 15).cuda()
#     # model = AdjustLayer(in_channels=256, out_channels=256)
#     model = Adjust_Transformer(256).cuda()
#     out = model(x)