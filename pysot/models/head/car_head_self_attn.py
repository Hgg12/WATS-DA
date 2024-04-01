import math
import torch
from torch import nn, Tensor
from torch.nn import init
from einops import rearrange

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        torch.nn.init.normal_(self.fc1.weight, std=0.01)
        torch.nn.init.normal_(self.fc2.weight, std=0.01)

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

        torch.nn.init.normal_(self.conv1.weight, std=0.01)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mask = torch.cat([avg_out, max_out], dim=1)
        mask = self.conv1(mask)
        mask = self.sigmoid(mask)

        out = x * mask
        return out


class AttentionBlock(nn.Module):
    def __init__(self, query_dim: int, n_heads: int, dim_head: int, context_dim: int = None, dropout: float = 0.0) -> None:
        super().__init__()

        self.n_heads = n_heads
        self.scale = dim_head ** -0.5

        inner_dim = dim_head * n_heads
        if context_dim is None:
            context_dim = query_dim

        self.to_q = nn.Conv2d(query_dim, inner_dim, kernel_size=1, stride=1, padding=0)
        self.to_k = nn.Conv2d(context_dim, inner_dim, kernel_size=1, stride=1, padding=0)
        self.to_v = nn.Conv2d(context_dim, inner_dim, kernel_size=1, stride=1, padding=0)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, query_dim, kernel_size=1, stride=1, padding=0),
            nn.Dropout(dropout)
        )

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: Tensor, context: Tensor = None) -> None:
        if context is None: context = x
        b, c, h, w = x.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        q, k, v = map(lambda _it: rearrange(_it, "b (n d) h w -> (b n) (h w) d", n=self.n_heads), (q, k, v))

        sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        sim = torch.softmax(sim, dim=-1)

        out = torch.einsum("b i j, b j d -> b i d", sim, v)
        out = rearrange(out, "(b n) (h w) d -> b (n d) h w", n=self.n_heads, h=h)
        out = self.to_out(out)
        return out


class CARHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.TRAIN.NUM_CLASSES

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.TRAIN.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        n_heads = 4
        self.sa = nn.ModuleList([AttentionBlock(in_channels, n_heads, in_channels // n_heads, dropout=0.15) for _ in range(cfg.TRAIN.NUM_CONVS)])
        self.ca = nn.ModuleList([AttentionBlock(in_channels, n_heads, in_channels // n_heads, dropout=0.15) for _ in range(cfg.TRAIN.NUM_CONVS)])

        self.add_module('cls_tower', nn.ModuleList(cls_tower))
        self.add_module('bbox_tower', nn.ModuleList(bbox_tower))
        
        # self.add_module('cls_tower', nn.Sequential(*cls_tower))
        # self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        torch.nn.init.constant_(l.bias, 0)
          
        # initialize the bias for focal loss
        prior_prob = cfg.TRAIN.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        fmap_cls = x
        for blk_idx, cls_blk in enumerate(self.cls_tower):
            fmap_cls = cls_blk(fmap_cls)
            if isinstance(cls_blk, nn.ReLU):
                fmap_cls = self.ca[blk_idx // 3](fmap_cls) + fmap_cls
        # cls_tower = self.cls_tower(x)
        # fmap_cls = self.ca(fmap_cls) + fmap_cls

        logits = self.cls_logits(fmap_cls)
        centerness = self.centerness(fmap_cls)

        fmap_box = x
        for blk_idx, bbox_blk in enumerate(self.bbox_tower):
            fmap_box = bbox_blk(fmap_box)
            if isinstance(cls_blk, nn.ReLU):
                fmap_box = self.sa[blk_idx // 3](fmap_box) + fmap_box
        # box_tower = self.bbox_tower(x)
        # fmap_box = self.sa(fmap_box) + fmap_box
        bbox_reg = torch.exp(self.bbox_pred(fmap_box))

        return logits, bbox_reg, centerness


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

