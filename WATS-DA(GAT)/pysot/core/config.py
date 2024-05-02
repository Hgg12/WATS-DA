# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "siamgat_googlenet"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 287

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.RESUME_D = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot_WATS-DA'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 16 # 1

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 3.0

__C.TRAIN.CEN_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.BASE_LR_d = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.TRAIN.NUM_CLASSES = 2

__C.TRAIN.NUM_CONVS = 4

__C.TRAIN.PRIOR_PROB = 0.01

__C.TRAIN.LOSS_ALPHA = 0.25

__C.TRAIN.LOSS_GAMMA = 2.0

__C.TRAIN.CHANNEL_NUM = 256

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

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

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

#__C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB', 'GOT', 'LaSOT', 'TrackingNet')
__C.DATASET.SOURCE = ['TrackingNet'] # only train on COCO for test  'VID', , 'DET', 'YOUTUBEBB' , 'COCO', 'YOUTUBEBB'
__C.DATASET.TARGET = ['WATB400_1','WATB400_2']  # only train on COCO for test  'VID', , 'DET', 'YOUTUBEBB'

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

__C.DATASET.WATB2024_3= CN()
__C.DATASET.WATB2024_3.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/WATB2024_3/crop511'
__C.DATASET.WATB2024_3.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/WATB2024_3/result/hqsam_WATB2024_3.json'
__C.DATASET.WATB2024_3.FRAME_RANGE = 1
__C.DATASET.WATB2024_3.NUM_USE = -1 # 10000

__C.DATASET.Wildlife2024_4_1= CN()
__C.DATASET.Wildlife2024_4_1.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/Wildlife2024_4.1/crop511'
__C.DATASET.Wildlife2024_4_1.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/Wildlife2024_4.1/result/sam_Wildlife2024_4.1.json'
__C.DATASET.Wildlife2024_4_1.FRAME_RANGE = 1
__C.DATASET.Wildlife2024_4_1.NUM_USE = -1 # 10000

__C.DATASET.Wildlife2024_4_2= CN()
__C.DATASET.Wildlife2024_4_2.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/Wildlife2024_4.2/crop511'
__C.DATASET.Wildlife2024_4_2.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/Wildlife2024_4.2/result/sam_Wildlife2024_4.2.json'
__C.DATASET.Wildlife2024_4_2.FRAME_RANGE = 1
__C.DATASET.Wildlife2024_4_2.NUM_USE = -1 # 10000
__C.DATASET.VIDEOS_PER_EPOCH = 20000
#__C.DATASET.VIDEOS_PER_EPOCH_B = 20000

# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

# Backbone type, current only support googlenet;alexnet;
__C.BACKBONE.TYPE = 'googlenet'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train backbone layers
__C.BACKBONE.TRAIN_LAYERS = []

# Train channel_layer
__C.BACKBONE.CHANNEL_REDUCE_LAYERS = []

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Crop_pad
__C.BACKBONE.CROP_PAD = 4

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 0

# Backbone offset
__C.BACKBONE.OFFSET = 13

# Backbone stride
__C.BACKBONE.STRIDE = 8

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Align layer options
# ------------------------------------------------------------------------ #
__C.ALIGN = CN()

# Adjust layer type
__C.ADJUST.TYPE = "GoogLeNetAdjustLayer"

# Align layer
__C.ALIGN.ALIGN = True

__C.ALIGN.KWARGS = CN(new_allowed=True)

# ALIGN layer type
__C.ALIGN.TYPE = "EFM"


# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

# SiamGAT
__C.TRAIN.ATTENTION = True

__C.TRACK.TYPE = 'SiamGATTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 287

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8

__C.TRACK.OFFSET = 45

__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44

# ------------------------------------------------------------------------ #
# HP_SEARCH parameters
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.OTB100 = [0.28, 0.16, 0.4]

# __C.HP_SEARCH.OTB100 = [0.32, 0.3, 0.38]

__C.HP_SEARCH.GOT_10k = [0.7, 0.02, 0.35]

# __C.HP_SEARCH.GOT_10k = [0.9, 0.25, 0.35]

__C.HP_SEARCH.UAV123 = [0.24, 0.04, 0.04]

__C.HP_SEARCH.UAVDark70 = [0.35, 0.05, 0.18] #0.24, 0.04, 0.04

__C.HP_SEARCH.NAT = [0.24, 0.04, 0.04] #0.24, 0.04, 0.04 #0.35, 0.05, 0.18

__C.HP_SEARCH.NUT_L = [0.24, 0.04, 0.04]

__C.HP_SEARCH.SONAT = [0.24, 0.04, 0.04]

__C.HP_SEARCH.LaSOT = [0.35, 0.05, 0.18]

__C.HP_SEARCH.WATB = [0.39, 0.04, 0.37] 

# __C.HP_SEARCH.LaSOT = [0.45, 0.05, 0.18]

# __C.HP_SEARCH.TrackingNet = [0.4, 0.05, 0.4]