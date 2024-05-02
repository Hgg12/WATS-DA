"""
This file contains specific functions for computing losses of SiamCAR
file
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import math


INF = 100000000


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero(as_tuple =False).squeeze().cuda()
    neg = label.data.eq(0).nonzero(as_tuple =False).squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()
# class IOULoss(nn.Module):
#     def __init__(self):
#         super(IOULoss, self).__init__()
#         self.loc_loss_type = 'linear_iou'
#     def forward(self, pred, target, weight=None):
#         pred_left = pred[:, 0]
#         pred_top = pred[:, 1]
#         pred_right = pred[:, 2]
#         pred_bottom = pred[:, 3]

#         target_left = target[:, 0]
#         target_top = target[:, 1]
#         target_right = target[:, 2]
#         target_bottom = target[:, 3]

#         pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
#         target_area = (target_left + target_right) * (target_top + target_bottom)

#         w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
#         g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
#         h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
#         g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
#         center_b_x = (pred_left+pred_right)/2
#         center_b_y = (pred_top+pred_bottom)/2
#         center_gtb_x = (target_left+target_right)/2
#         center_gtb_y = (target_top+target_bottom)/2
#         center_distance = (center_gtb_x-center_b_x)**2 + (center_gtb_y-center_b_y)**2
#         c_distance = g_w_intersect**2 + g_h_intersect**2
#         ac_uion = g_w_intersect * g_h_intersect + 1e-7
#         area_intersect = w_intersect * h_intersect
#         area_union = target_area + pred_area - area_intersect
#         ious = (area_intersect + 1.0) / (area_union + 1.0)
#         gious = ious - (ac_uion - area_union) / ac_uion
#         DIOU = ious - center_distance /c_distance

#         if self.loc_loss_type == 'iou':
#             losses = -torch.log(ious)
#         elif self.loc_loss_type == 'linear_iou':
#             losses = 1 - ious
#         elif self.loc_loss_type == 'giou':
#             losses = 1 - gious
#         elif self.loc_loss_type == 'DIOU':
#             losses = 1 - DIOU
#         else:
#             raise NotImplementedError

#         if weight is not None and weight.sum() > 0:
#             return (losses * weight).sum() / weight.sum()
#         else:
#             assert losses.numel() != 0
#             return losses.mean()
        
# class CIOULoss(nn.Module):
#     def forward(self, pred, target, weight=None):
#         pred_left = pred[:, 0]
#         pred_top = pred[:, 1]
#         pred_right = pred[:, 2]
#         pred_bottom = pred[:, 3]

#         target_left = target[:, 0]
#         target_top = target[:, 1]
#         target_right = target[:, 2]
#         target_bottom = target[:, 3]

#         target_area = (target_right - target_left) * (target_bottom - target_top)
#         pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)

#         w_intersect = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
#         h_intersect = torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)

#         area_intersect = w_intersect * h_intersect
#         area_union = target_area + pred_area - area_intersect

#         # 计算中心点坐标
#         x1_c = (pred_left + pred_right) / 2
#         y1_c = (pred_top + pred_bottom) / 2
#         x2_c = (target_left + target_right) / 2
#         y2_c = (target_top + target_bottom) / 2

#         # 计算对角线距离的平方
#         di_square = (x1_c - x2_c)**2 + (y1_c - y2_c)**2

#         # 计算CIOU修正项
#         cioU = (area_intersect + 1.0) / (area_union + 1.0)
#         diou_term = di_square / ((pred_right - pred_left)**2 + (pred_bottom - pred_top)**2 +
#                                  (target_right - target_left)**2 + (target_bottom - target_top)**2 - di_square + 1e-7)

#         cioU_loss = -torch.log(cioU + 1e-7) + diou_term
#         cioU_loss=1-cioU_loss
#         if weight is not None and weight.sum() > 0:
#             return (cioU_loss * weight).sum() / weight.sum()
#         else:
#             assert cioU_loss.numel() != 0
#             return cioU_loss.mean()


# class EIOULoss(nn.Module):
#     def forward(self, pred, target, weight=None, GIoU=False, DIoU=False, CIoU=False,  EIoU=True, eps=1e-7):
#         # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
#         box2 = target.T
#         box1=pred
#         # Get the coordinates of bounding boxes
       
#         b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
#         b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
#         # b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
#         # b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
#         # b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
#         # b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    
#         # Intersection area
#         inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
#                 (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
#         # Union Area
#         w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
#         w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
#         union = w1 * h1 + w2 * h2 - inter + eps
    
#         iou = inter / union
#         if GIoU or DIoU or CIoU or EIoU:
#             cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
#             ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
#             if CIoU or DIoU or EIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
#                 c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
#                 rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
#                         (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
#                 if DIoU:
#                     return iou - rho2 / c2  # DIoU
#                 elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
#                     v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
#                     with torch.no_grad():
#                         alpha = v / (v - iou + (1 + eps))
#                     return iou - (rho2 / c2 + v * alpha)  # CIoU
#                 elif EIoU:
#                     rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
#                     rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
#                     cw2 = cw ** 2 + eps
#                     ch2 = ch ** 2 + eps
#                     losses=1-(iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2))
#                     if weight is not None and weight.sum() > 0:
#                         return (losses * weight).sum() / weight.sum()
#                     else:
#                         assert losses.numel() != 0
#                         return losses.mean()
#             else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
#                 c_area = cw * ch + eps  # convex area
#                 return iou - (c_area - union) / c_area  # GIoU
#         else:
#             return iou  # IoU



# class AlphaIOULoss(nn.Module):
#     def __init__(self, alpha=3):
#         super(AlphaIOULoss, self).__init__()
#         self.alpha = alpha

#     def forward(self, pred, target, weight=None):
#         pred_left = pred[:, 0]
#         pred_top = pred[:, 1]
#         pred_right = pred[:, 2]
#         pred_bottom = pred[:, 3]

#         target_left = target[:, 0]
#         target_top = target[:, 1]
#         target_right = target[:, 2]
#         target_bottom = target[:, 3]

#         target_area = (target_right - target_left) * (target_bottom - target_top)
#         pred_area = (pred_right - pred_left) * (pred_bottom - pred_top)

#         w_intersect = torch.min(pred_right, target_right) - torch.max(pred_left, target_left)
#         h_intersect = torch.min(pred_bottom, target_bottom) - torch.max(pred_top, target_top)

#         area_intersect = w_intersect * h_intersect
#         area_union = target_area + pred_area - area_intersect

#         # 计算中心点坐标
#         x1_c = (pred_left + pred_right) / 2
#         y1_c = (pred_top + pred_bottom) / 2
#         x2_c = (target_left + target_right) / 2
#         y2_c = (target_top + target_bottom) / 2

#         # 计算对角线距离的平方
#         di_square = (x1_c - x2_c)**2 + (y1_c - y2_c)**2

#         # 计算IoU损失
#         iou = area_intersect / (area_union + 1e-7)

#         # 计算CIOU修正项
#         cioU = 1 - ((di_square + 1e-7) / (di_square + area_union + 1e-7))

#         # 计算αIOU损失
#         alpha_iou_loss = (1 - iou + cioU).clamp(min=0)

#         if weight is not None and weight.sum() > 0:
#             return (alpha_iou_loss * weight).sum() / weight.sum()
#         else:
#             assert alpha_iou_loss.numel() != 0
#             return alpha_iou_loss.mean()


# class SIOULoss(nn.Module):
#     def forward(self, pred, target, weight=None):
       
#         weight=None
#         # Calculate intersection coordinates
#         x1 = torch.max(pred[:, 0].unsqueeze(1), target[:, 0].unsqueeze(0))
#         y1 = torch.max(pred[:, 1].unsqueeze(1), target[:, 1].unsqueeze(0))
#         x2 = torch.min(pred[:, 2].unsqueeze(1), target[:, 2].unsqueeze(0))
#         y2 = torch.min(pred[:, 3].unsqueeze(1), target[:, 3].unsqueeze(0))
    
#         # Calculate intersection area
#         intersection_area = torch.clamp(x2 - x1 + 1, min=0) * torch.clamp(y2 - y1 + 1, min=0)
    
#         # Calculate box areas
#         boxes_a_area = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
#         boxes_b_area = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
    
#         # Calculate union area
#         union_area = boxes_a_area.unsqueeze(1) + boxes_b_area.unsqueeze(0) - intersection_area
    
#         # Calculate IoU
#         iou = intersection_area / union_area
    
#         # Compute SIoU terms
#         si = torch.min(pred[:, 2], target[:, 2]) - torch.max(pred[:, 0], target[:, 0]) + 1
#         sj = torch.min(pred[:, 3], target[:, 3]) - torch.max(pred[:, 1], target[:, 1]) + 1
#         s_union = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1) + \
#                 (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
#         s_intersection = si * sj
    
#         # Compute SCYLLA-IoU
#         siou = iou - (s_intersection / s_union)
#         losses=1-siou
#     # Compute loss
#         if weight is not None and weight.sum() > 0:
#             b = losses.shape[0]
#             weight=weight.view(1,b).repeat(b,1)
#             return (losses * weight).sum() / weight.sum()
#         else:
#             assert losses.numel() != 0
#             return losses.mean()


# class WIOULoss(nn.Module):
#     def forward(self, pred, target, weight=None):
        
#         """
#         Compute the Weighted IoU loss between predicted and target bounding boxes.
#         其中，输入pred_boxes和target_boxes分别是形状为(N, 4)的预测边界框和目标边界框张量。
#         如果需要使用权重，则输入形状为(N,)的权重张量weight，否则默认为None。函数返回一个标量，表示计算出的加权IoU损失。
#         Args:
#             pred_boxes (torch.Tensor): Predicted bounding boxes, with shape (N, 4).
#             target_boxes (torch.Tensor): Target bounding boxes, with shape (N, 4).
#             weight (torch.Tensor, optional): Weight tensor with shape (N,). Defaults to None.
#         Returns:
#             torch.Tensor: Weighted IoU loss scalar.
#         """
#         # Compute the intersection over union (IoU) between the predicted and target boxes.
#         x1 = torch.max(pred[:, 0], target[:, 0])
#         y1 = torch.max(pred[:, 1], target[:, 1])
#         x2 = torch.min(pred[:, 2], target[:, 2])
#         y2 = torch.min(pred[:, 3], target[:, 3])
#         intersection_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
#         pred_boxes_area = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
#         target_boxes_area = (target[:, 2] - target[:, 0]) * (target[:, 3] - target[:, 1])
#         union_area = pred_boxes_area + target_boxes_area - intersection_area
#         iou = intersection_area / union_area
#         # Compute the Weighted IoU loss using the IoU and weight tensors.
#         if weight is None:
#             w_iou = 1 - iou.mean()
#         else:
#             w_iou = (1 - iou) * weight
#             w_iou = w_iou.sum() / weight.sum()
    
#         return w_iou

#         # if weight is not None and weight.sum() > 0:
#         #     return (wiou_loss * weight).sum() / weight.sum()
#         # else:
#         #     assert wiou_loss.numel() != 0
#         #     return wiou_loss.mean()



class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self,cfg):
        self.box_reg_loss_func = IOULoss()
        # self.box_reg_loss_func = CIOULoss()
        # self.box_reg_loss_func = EIOULoss()
        # self.box_reg_loss_func = AlphaIOULoss()
        # self.box_reg_loss_func = SIOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, labels, gt_bbox):
        # reg_targets = []
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE**2,-1)

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        s1 = reg_targets_per_im[:, :, 0] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1

        return labels.permute(1,0).contiguous(), reg_targets_per_im.permute(1,0,2).contiguous()

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """

        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets)
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss


def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator
