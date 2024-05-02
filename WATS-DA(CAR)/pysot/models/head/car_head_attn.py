import torch
from torch import nn
import math


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        mask = avg_out + max_out
        mask = self.sigmoid(mask)
        
        out = x * mask
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 7,3     3,1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_out, max_out], dim=1)
        mask = self.conv1(mask)
        mask = self.sigmoid(mask)

        out = x * mask
        return out


class ConvNormRelu(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, channels),
            nn.ReLU())

    def forward(self, x):
        return self.blk(x)


class CARHead_Attn(torch.nn.Module):
    def __init__(self, cfg, channels):
        super(CARHead_Attn, self).__init__()
        num_classes = cfg.TRAIN.NUM_CLASSES

        self.cls_tower = nn.ModuleList()
        for i in range(cfg.TRAIN.NUM_CONVS):
            self.cls_tower.append(ConvNormRelu(channels))
            # self.cls_tower.append(ChannelAttention(channels))

        self.bbox_tower = nn.ModuleList()
        for i in range(cfg.TRAIN.NUM_CONVS):
            self.bbox_tower.append(ConvNormRelu(channels))
            # self.bbox_tower.append(SpatialAttention())

        self.cls_logits = nn.Conv2d(channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1)
        self.bbox_regression = nn.Conv2d(channels, 4, kernel_size=3, stride=1, padding=1)
        
        self.reset_parameters(cfg.TRAIN.PRIOR_PROB)

    def reset_parameters(self, prior_prob):
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_regression, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)
                elif isinstance(l, ConvNormRelu):
                    for _blk in l.modules():
                        if isinstance(_blk, nn.Conv2d):
                            torch.nn.init.normal_(_blk.weight, std=0.01)
                            torch.nn.init.constant_(_blk.bias, 0)         

        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)        

    def forward(self, x):
        fmap_cls = x
        for cls_blk in self.cls_tower:
            fmap_cls = cls_blk(fmap_cls) #+ fmap_cls

        logits = self.cls_logits(fmap_cls)
        centerness = self.centerness(fmap_cls)

        fmap_bbox = x
        for bbox_blk in self.bbox_tower:
            fmap_bbox = bbox_blk(fmap_bbox) #+ fmap_bbox

        bbox_reg = torch.exp(self.bbox_regression(fmap_bbox))

        return logits, bbox_reg, centerness


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

