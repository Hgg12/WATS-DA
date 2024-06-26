import torch.nn as nn
import torch
from ltr.external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from torch.nn import functional as F
from ltr.models.neck.neck_utils import *

# from torch2trt import torch2trt

class CorrNL(nn.Module):
    """Network module for IoU prediction. Refer to the ATOM paper for an illustration of the architecture.
    It uses two backbone feature layers as input.
    args:
        input_dim:  Feature dimensionality of the two input backbone layers.
        pred_input_dim:  Dimensionality input the the prediction network.
        pred_inter_dim:  Intermediate dimensionality in the prediction network."""

    def __init__(self, pool_size=8, use_post_corr=True, use_NL=True):
        super().__init__()
        '''PrRoIPool2D的第三个参数是当前特征图尺寸相对于原图的比例(下采样率)'''
        '''layer2的stride是8, layer3的stride是16
        当输入分辨率为256x256时, layer3的输出分辨率为16x16, 目标尺寸大约为8x8
        ##### 注意: 如果输入分辨率改变,或者使用的层改变,那么这块的参数需要重新填 #####'''
        self.prroi_pool = PrRoIPool2D(pool_size, pool_size, 1/16)
        num_corr_channel = pool_size*pool_size
        self.use_post_corr = use_post_corr
        if use_post_corr:
            self.post_corr = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        self.channel_attention = SEModule(num_corr_channel,reduction=4)
        self.use_NL = use_NL
        if self.use_NL is True:
            self.spatial_attention = NONLocalBlock2D(in_channels=num_corr_channel)
        elif self.use_NL is False:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(64),
            )
        else:
            self.spatial_attention = nn.Sequential()
    def forward(self, feat1, feat2, bb1):
        """Runs the ATOM IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.
        args:
            feat1:  Features from the reference frames (4 or 5 dims).
            feat2:  Features from the test frames (4 or 5 dims).
            bb1:  Target boxes (x,y,w,h) in image coords in the reference samples. Dims (images, sequences, 4).
            proposals2:  Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4)."""

        assert bb1.dim() == 3
        # num_images, num_sequences = bb1.size()[:2] # 1, 64

        # Extract first train sample
        if len(feat1)==1:
            feat1 = feat1[0] # size为(64,C,H,W)
            feat2 = feat2[0] # size为(64,C,H,W)
            bb1 = bb1[0,...] # (64,4)
        else:
            raise ValueError("目前只支持使用单层特征图")
        '''get PrRoIPool feature '''
        # Add batch_index to rois
        batch_size = bb1.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb1.device) # (64,1)
        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb1 = bb1.clone()
        bb1[:, 2:4] = bb1[:, 0:2] + bb1[:, 2:4]
        roi1 = torch.cat((batch_index, bb1), dim=1) #(64,1),(64,4) ---> (64,5)
        feat_roi1 = self.prroi_pool(feat1, roi1) # (64,C,H,W)
        feat_corr,_ = self.corr_fun(feat_roi1, feat2)
        # print('相关后的特征维度是:',feat_corr.size())#(batch,StxSt,Sr,Sr)
        '''channel attention: Squeeze and Excitation'''
        feat_ca = self.channel_attention(feat_corr) # 计算通道注意力特征
        '''spatial attention: Non-local 2D'''
        feat_sa = self.spatial_attention(feat_ca)
        return feat_sa

    def get_ref_kernel(self, feat1, bb1):
        assert bb1.dim() == 3
        # num_images, num_sequences = bb1.size()[:2] # 1, 64

        # Extract first train sample
        if len(feat1) == 1:
            feat1 = feat1[0]  # size为(64,C,H,W)
            bb1 = bb1[0, ...]  # (64,4)
        else:
            raise ValueError("目前只支持使用单层特征图")
        '''get PrRoIPool feature '''
        # Add batch_index to rois
        batch_size = bb1.shape[0]
        batch_index = torch.arange(batch_size, dtype=torch.float32).view(-1, 1).to(bb1.device)  # (64,1)
        # input bb is in format xywh, convert it to x0y0x1y1 format
        bb1 = bb1.clone()
        bb1[:, 2:4] = bb1[:, 0:2] + bb1[:, 2:4]
        roi1 = torch.cat((batch_index, bb1), dim=1)  # (64,1),(64,4) ---> (64,5)
        '''注意: feat1和roi1必须都是cuda tensor,不然会计算失败(不报错但是其实cuda已经出现问题了,会影响后续的计算)'''
        self.ref_kernel = self.prroi_pool(feat1, roi1)  # (64,C,H,W)

    def fuse_feat(self, feat2, convert_trt=False):
        '''fuse features from reference and test branch'''
        if len(feat2) == 1:
            feat2 = feat2[0]
        '''Step1: pixel-wise correlation'''
        feat_corr,_ = self.corr_fun(self.ref_kernel, feat2)

        # zxy for debugging
        # import devtool
        # import cv2
        # response_maps = feat_corr[0]
        #
        # for idx, response_map in enumerate(response_maps):
        #     response_map = devtool.vis.imagesc(response_map)
        #     response_map = cv2.resize(response_map, (256, 256), interpolation=cv2.INTER_NEAREST)
        #     # devtool.vis.imshow(response_map, default_path='/home/zxy/Desktop/ptw_corr')
        #     devtool.vis.imwrite(response_map, name="{:03}.jpg".format(idx+1), default_dir='/home/zxy/Desktop/vis/pw_corr')

        # zxy for debugging
        # import devtool.node_buffer
        # norm_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, None, :]
        # norm_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, None, :]
        # train_imgs = devtool.node_buffer.train_imgs
        # test_imgs = devtool.node_buffer.test_imgs
        # train_img = train_imgs[0][0].permute(1, 2, 0).cpu()*norm_std + norm_mean
        # test_img = test_imgs[0][0].permute(1, 2, 0).cpu()*norm_std + norm_mean
        # train_img, test_img = train_img*255, test_img*255
        # devtool.vis.imwrite(train_img, name="ref_img.jpg", default_dir='/home/zxy/Desktop/vis')
        # devtool.vis.imwrite(test_img, name="search_region_img.jpg", default_dir='/home/zxy/Desktop/vis')

        # print('相关后的特征维度是:',feat_corr.size())#(batch,StxSt,Sr,Sr) (batch,64,16,16)
        '''Step2: channel attention: Squeeze and Excitation'''
        if self.use_post_corr:
            if not convert_trt:
                feat_corr = self.post_corr(feat_corr)
            else:
                # tensorrt transform part 2
                post_corr_trt = torch2trt(self.post_corr, [feat_corr])
                feat_corr = post_corr_trt(feat_corr)
                torch.save(post_corr_trt.state_dict(), 'trt_models/trt_model_RF_RF_efb0/post_corr_trt.pth')
                print('**********************TensorRT model 2 convert complete**********************')
        if not convert_trt:
            feat_ca = self.channel_attention(feat_corr) # 计算通道注意力特征
        else:
            # tensorrt transform part 3
            channel_attention_trt = torch2trt(self.channel_attention, [feat_corr])
            feat_ca = channel_attention_trt(feat_corr)
            torch.save(channel_attention_trt.state_dict(), 'trt_models/trt_model_RF_RF_efb0/channel_attention_trt.pth')
            print('**********************TensorRT model 3 convert complete**********************')
        if self.use_NL is False:
            # print('not use non-local')
            feat_ca = feat_ca + self.spatial_attention(feat_ca)
            return feat_ca
        else:
            '''Step3: spatial attention: Non-local 2D'''
            feat_sa = self.spatial_attention(feat_ca)

            # zxy for debugging
            # import devtool; import cv2
            # sp_att = devtool.node_buffer.sp_atts[0]
            # for idx, att in enumerate(sp_att):
            #     att = att.reshape(8, 8)
            #     att = devtool.vis.imagesc(att)
            #     att = cv2.resize(att, (256, 256), interpolation=cv2.INTER_NEAREST)
            #     # devtool.vis.imshow(att, default_path='/home/zxy/Desktop/ptw_corr')
            #     devtool.vis.imwrite(att, name="{:03}.jpg".format(idx + 1),
            #                         default_dir='/home/zxy/Desktop/vis/nl_att')

            return feat_sa

    def corr_fun(self, Kernel_tmp, Feature, KERs=None):
        size = Kernel_tmp.size()
        CORR = []
        Kernel = []
        for i in range(len(Feature)):
            ker = Kernel_tmp[i:i + 1]
            fea = Feature[i:i + 1]
            ker = ker.view(size[1], size[2] * size[3]).transpose(0, 1)
            ker = ker.unsqueeze(2).unsqueeze(3)
            if not (type(KERs) == type(None)):
                ker = torch.cat([ker, KERs[i]], 0)
            co = F.conv2d(fea, ker.contiguous())
            CORR.append(co)
            ker = ker.unsqueeze(0)
            Kernel.append(ker)
        corr = torch.cat(CORR, 0)
        Kernel = torch.cat(Kernel, 0)
        return corr, Kernel

    def fuse_feat_trt(self, feat2):
        '''fuse features from reference and test branch'''
        if len(feat2) == 1:
            feat2 = feat2[0]
        '''Step1: pixel-wise correlation'''
        feat_corr,_ = self.corr_fun(self.ref_kernel, feat2)
        return feat_corr
