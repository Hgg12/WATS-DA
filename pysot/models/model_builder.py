# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pysot.core.config import cfg
from pysot.models.loss_car import make_siamcar_loss_evaluator
from pysot.models.backbone import get_backbone
from pysot.models.head.car_head import CARHead
# from pysot.models.head.car_head_erh import CARHead
# from pysot.models.head.car_head__ import CARHead
# from pysot.models.head.car_head_attn import CARHead_Attn
# from pysot.models.head.car_head_self_attn import CARHead
# from pysot.models.head.car_head_double import CARHead
from pysot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from pysot.utils.xcorr import xcorr_depthwise
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# class Rank_CLS_Loss(nn.Module):
#     def __init__(self, L=4, margin=0.5):
#         super(Rank_CLS_Loss, self).__init__()
#         self.margin =margin
#         self.L = L

#     def forward(self,input, label):
#         loss_all = []
#         batch_size=input.shape[0]
#         pred=input.view(batch_size,-1,2)
#         label =label.view(batch_size,-1)
#         for batch_id in range(batch_size):
#             pos_index = np.where(label[batch_id].cpu() == 1)[0].tolist()
#             neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
#             if len(pos_index) > 0:
#                pos_prob = torch.exp(pred[batch_id][pos_index][:,1])
#                neg_prob = torch.exp(pred[batch_id][neg_index][:,1])
        
#                num_pos=len(pos_index)
#                neg_value, _ = neg_prob.sort(0, descending=True)
#                pos_value,_ =pos_prob.sort(0,descending=True)
#                neg_idx2=neg_prob>cfg.TRAIN.HARD_NEGATIVE_THS
#                if neg_idx2.sum()==0:
#                    continue
#                neg_value=neg_value[0:num_pos]
        
#                pos_value=pos_value[0:num_pos]
#                neg_q = F.softmax(neg_value, dim=0)
#                neg_dist = torch.sum(neg_value*neg_q)
            
#                pos_dist = torch.sum(pos_value)/len(pos_value)
#                loss = torch.log(1.+torch.exp(self.L*(neg_dist - pos_dist+self.margin)))/self.L
#             else:
#                neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
#                neg_prob = torch.exp(pred[batch_id][neg_index][:,1])
#                neg_value, _ = neg_prob.sort(0, descending=True)
#                neg_idx2=neg_prob>cfg.TRAIN.HARD_NEGATIVE_THS
#                if neg_idx2.sum()==0:
#                     continue
#                num_neg=len(neg_prob[neg_idx2])
#                num_neg=max(num_neg,cfg.TRAIN.RANK_NUM_HARD_NEGATIVE_SAMPLES)
#                neg_value=neg_value[0:num_neg]
#                neg_q = F.softmax(neg_value, dim=0)
#                neg_dist = torch.sum(neg_value*neg_q)
#                loss = torch.log(1.+torch.exp(self.L*(neg_dist - 1. + self.margin)))/self.L
               
#             loss_all.append(loss)
#         if len(loss_all):
#             final_loss = torch.stack(loss_all).mean()
#         else:
#             final_loss=torch.zeros(1).cuda()
           
#         return final_loss

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS).cuda()

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        if cfg.ALIGN.ALIGN:
            self.align = get_neck(cfg.ALIGN.TYPE,
                                 **cfg.ALIGN.KWARGS)

        # build car head
        # self.car_head = CARHead(cfg, 256)
        self.car_head = CARHead(cfg, 256)
        #
        # self.rank_cls_loss=Rank_CLS_Loss()
        # self.rank_loc_loss=rank_loc_loss()
        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        # if cfg.ALIGN.ALIGN:
            # zf = [self.align(zf[i]) for i in range(len(zf))]
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        if cfg.ALIGN.ALIGN:
            # xf = [self.align(xf[i]) for i in range(len(xf))]
            xf = [self.align(_zf, _xf) for _zf, _xf in zip(self.zf, xf)]

        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['bbox'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        if cfg.ALIGN.ALIGN:
            # zf = [self.align(_zf) for _zf in zf]
            # xf = [self.align(_xf) for _xf in xf]
            xf = [self.align(_zf, _xf) for _zf, _xf in zip(zf, xf)]
        features = self.xcorr_depthwise(xf[0],zf[0])
        
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)

        features = self.down(features)

        cls, loc, cen = self.car_head(features)

        # loc = torch.clamp(loc, min=0.)

        locations = compute_locations(cls, cfg.TRACK.STRIDE)
        cls = self.log_softmax(cls)
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss
        # CR_loss=self.rank_cls_loss(cls,label_cls)
        # IGR_loss_1,IGR_loss_2=self.rank_loc_loss(cls,label_cls,locations,label_target)
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss 
        # + cfg.TRAIN.RANK_CLS_WEIGHT*CR_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        # outputs['CR_loss']= CR_loss
        return outputs, zf, xf
