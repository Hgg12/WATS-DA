# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamban_r50_l234"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Number of negative
__C.TRAIN.NEG_NUM = 16

# Number of positive
__C.TRAIN.POS_NUM = 16

# Number of anchors per images
__C.TRAIN.TOTAL_NUM = 64


__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.RESUME_D = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 1.0

__C.TRAIN.RANK_CLS_WEIGHT = 0.5

__C.TRAIN.RANK_IGR_WEIGHT = 0.25

__C.TRAIN.HARD_NEGATIVE_THS = 0.5

__C.TRAIN.RANK_NUM_HARD_NEGATIVE_SAMPLES = 8

__C.TRAIN.IoU_Gamma = 3

__C.TRAIN.HARD_NEGATIVE_THS = 0.5



__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.0015

__C.TRAIN.BASE_LR_d = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# Random shift see [SiamPRN++](https://arxiv.org/pdf/1812.11703)
# for detail discussion
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# Sample Negative pair see [DaSiamRPN](https://arxiv.org/pdf/1808.06048)
# for detail discussion
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

# __C.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'COCO', 'GOT10K', 'NAT') # 'DET',  'LASOT'
__C.DATASET.SOURCE = ['TrackingNet'] # only train on COCO for test  'VID', , 'DET', 'YOUTUBEBB' , 'COCO', 'YOUTUBEBB'
__C.DATASET.TARGET = ['WATB400_1','WATB400_2','Wildlife2024_7_1']  # only train on COCO for test  'VID', , 'DET', 'YOUTUBEBB'

__C.DATASET.VID = CN()
__C.DATASET.VID.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/vid/crop511'
__C.DATASET.VID.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/vid/train.json'
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE = -1 # 10000  # repeat until reach NUM_USE

__C.DATASET.TrackingNet = CN()
__C.DATASET.TrackingNet.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/TrackingNet/crop511'
__C.DATASET.TrackingNet.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/TrackingNet/train.json'
__C.DATASET.TrackingNet.FRAME_RANGE = 100
__C.DATASET.TrackingNet.NUM_USE = -1 # 10000  # repeat until reach NUM_USE

__C.DATASET.WATB400_1= CN()
__C.DATASET.WATB400_1.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/WATB400-1/crop511'
__C.DATASET.WATB400_1.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/WATB400-1/result/hqsam_WATB400-1.json'
__C.DATASET.WATB400_1.FRAME_RANGE = 1
__C.DATASET.WATB400_1.NUM_USE = -1 # 10000

__C.DATASET.WATB400_2= CN()
__C.DATASET.WATB400_2.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/WATB400-2/crop511'
__C.DATASET.WATB400_2.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/WATB400-2/result/hqsam_WATB400-2.json'
__C.DATASET.WATB400_2.FRAME_RANGE = 1
__C.DATASET.WATB400_2.NUM_USE = -1 # 10000

__C.DATASET.Wildlife2024_7_1= CN()
__C.DATASET.Wildlife2024_7_1.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/Wildlife2024_7_1/crop511'
__C.DATASET.Wildlife2024_7_1.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/Wildlife2024_7_1/result/Wildlife2024_7_1.json'
__C.DATASET.Wildlife2024_7_1.FRAME_RANGE = 1
__C.DATASET.Wildlife2024_7_1.NUM_USE = -1 # 10000


__C.DATASET.VIDEOS_PER_EPOCH = 20000
__C.DATASET.VIDEOS_PER_EPOCH_B = 20000
__C.DATASET.VIDEOS_PER_EPOCH_S = 10000
__C.DATASET.VIDEOS_PER_EPOCH_T = 6667
__C.DATASET.VIDEOS_PER_EPOCH_N = 4000
# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support resnet18,34,50;alexnet;mobilenet
__C.BACKBONE.TYPE = 'res50'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer2', 'layer3', 'layer4']

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# Align layer options
# ------------------------------------------------------------------------ #
__C.ALIGN = CN()

# Align layer
__C.ALIGN.ALIGN = False

__C.ALIGN.KWARGS = CN(new_allowed=True)

# ALIGN layer type
__C.ALIGN.TYPE = "EFM"

# ------------------------------------------------------------------------ #
# BAN options
# ------------------------------------------------------------------------ #
__C.BAN = CN()

# Whether to use ban head
__C.BAN.BAN = False

# BAN type
__C.BAN.TYPE = 'MultiBAN'

__C.BAN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Point options
# ------------------------------------------------------------------------ #
__C.POINT = CN()

# Point stride
__C.POINT.STRIDE = 8

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamBANTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.14

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.45

# Interpolation learning rate
__C.TRACK.LR = 0.30

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.NAT = [0.473, 0.02, 0.385]

__C.HP_SEARCH.NAT_L = [0.466, 0.083, 0.404]

__C.HP_SEARCH.UAVDark70 = [0.473, 0.06, 0.305]

__C.HP_SEARCH.UAVDark135 = [0.473, 0.06, 0.305]

__C.HP_SEARCH.DarkTrack2021 = [0.473, 0.06, 0.305]

__C.HP_SEARCH.NUT_L = [0.473, 0.02, 0.385]

__C.HP_SEARCH.WATB = [0.39, 0.04, 0.37] 

