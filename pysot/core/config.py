# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "udatcar_r50_l234"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Anchor Target
__C.TRAIN.EXEMPLAR_SIZE = 127

__C.TRAIN.SEARCH_SIZE = 255

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.RESUME_D = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs_wrandom'

__C.TRAIN.SNAPSHOT_DIR = './snapshot_wrandom'

__C.TRAIN.EPOCH = 20

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 32

__C.TRAIN.NUM_WORKERS = 8

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 2.0

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

__C.TRAIN.RANK_CLS_WEIGHT = 0.5

__C.TRAIN.HARD_NEGATIVE_THS = 0.5

__C.TRAIN.RANK_NUM_HARD_NEGATIVE_SAMPLES = 8

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
# __C.DATASET.SEARCH.SCALE = 0

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion
__C.DATASET.NEG = 0.0

__C.DATASET.GRAY = 0.0

# __C.DATASET.NAMES = ('VID', 'COCO', 'DET', 'YOUTUBEBB')

__C.DATASET.SOURCE = ['TrackingNet'] # only train on COCO for test  'VID', , 'DET', 'YOUTUBEBB' , 'COCO', 'YOUTUBEBB'
__C.DATASET.TARGET = ['WATB400_1','WATB400_2','WATB2024_3']  # only train on COCO for test  'VID', , 'DET', 'YOUTUBEBB'

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

__C.DATASET.YOUTUBEBB = CN()
__C.DATASET.YOUTUBEBB.ROOT = 'train_dataset/yt_bb/crop511'  # YOUTUBEBB dataset path
__C.DATASET.YOUTUBEBB.ANNO = 'train_dataset/yt_bb/train.json'
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3
__C.DATASET.YOUTUBEBB.NUM_USE = -1  # use all not repeat

__C.DATASET.COCO = CN()
__C.DATASET.COCO.ROOT = 'train_dataset/coco/crop511'         # COCO dataset path
__C.DATASET.COCO.ANNO = 'train_dataset/coco/train2017.json'
__C.DATASET.COCO.FRAME_RANGE = 1
__C.DATASET.COCO.NUM_USE = -1

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = 'train_dataset/det/crop511'           # DET dataset path
__C.DATASET.DET.ANNO = 'train_dataset/det/train.json'
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = -1

__C.DATASET.GOT10K = CN()
__C.DATASET.GOT10K.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/crop511'
__C.DATASET.GOT10K.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/got10k/train.json'
__C.DATASET.GOT10K.FRAME_RANGE = 50
__C.DATASET.GOT10K.NUM_USE = -1 # 100000

__C.DATASET.LaSOT = CN()
__C.DATASET.LaSOT.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/lasot/crop511'         # LaSOT dataset path
__C.DATASET.LaSOT.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/lasot/train.json'
__C.DATASET.LaSOT.FRAME_RANGE = 100
__C.DATASET.LaSOT.NUM_USE = 100000

__C.DATASET.DarkTrack = CN()
__C.DATASET.DarkTrack.ROOT = '/home/mist/v4r/train_dataset/random_train/random_crop511' # '/home/mist/v4r/train_dataset/NAT2021-train/crop_511'         # GOT dataset path
__C.DATASET.DarkTrack.ANNO = '/home/mist/v4r/train_dataset/random_train/train_random.json' # '/home/mist/v4r/train_dataset/NAT2021-train/train.json'
__C.DATASET.DarkTrack.FRAME_RANGE = 50
__C.DATASET.DarkTrack.NUM_USE = 10000 # 100000

__C.DATASET.NAT = CN()
__C.DATASET.NAT.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/sam_nat/crop511'
__C.DATASET.NAT.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/sam_nat/sam_nat_b.json'
__C.DATASET.NAT.ANNO_B = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/sam_nat/sam_nat_b.json'
__C.DATASET.NAT.ANNO_S = './train_dataset/sam_nat/sam_nat_s.json'
__C.DATASET.NAT.ANNO_T = './train_dataset/sam_nat/sam_nat_t.json'
__C.DATASET.NAT.ANNO_N = './train_dataset/sam_nat/sam_nat_n.json'
__C.DATASET.NAT.FRAME_RANGE = 1
__C.DATASET.NAT.NUM_USE = -1 # 10000

__C.DATASET.ExDark = CN()
__C.DATASET.ExDark.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/ExDark/crop511'
__C.DATASET.ExDark.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/ExDark/sam_exdark.json'
__C.DATASET.ExDark.FRAME_RANGE = 1
__C.DATASET.ExDark.NUM_USE = -1 # 10000

__C.DATASET.wild = CN()
__C.DATASET.wild.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/wild2/crop511'
__C.DATASET.wild.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/wild2/result/hqsam_wild.json'
__C.DATASET.wild.FRAME_RANGE = 1
__C.DATASET.wild.NUM_USE = -1 # 10000

__C.DATASET.WATB_50 = CN()
__C.DATASET.WATB_50.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/WATB_50/crop511'
__C.DATASET.WATB_50.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/WATB_50/train.json'
__C.DATASET.WATB_50.FRAME_RANGE = 1
__C.DATASET.WATB_50.NUM_USE = -1 # 10000



__C.DATASET.DarkFace = CN()
__C.DATASET.DarkFace.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/DarkFace/crop511'
__C.DATASET.DarkFace.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/DarkFace/result/sam_DarkFace.json'
__C.DATASET.DarkFace.FRAME_RANGE = 1
__C.DATASET.DarkFace.NUM_USE = -1 # 10000

__C.DATASET.Animals = CN()
__C.DATASET.Animals.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/TEST/crop511'
__C.DATASET.Animals.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/TEST/result/hqsam_animals.json'
__C.DATASET.Animals.FRAME_RANGE = 1
__C.DATASET.Animals.NUM_USE = -1 # 10000

__C.DATASET.UAV = CN()
__C.DATASET.UAV.ROOT = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/UAV/crop511'
__C.DATASET.UAV.ANNO = '/media/w/719A549756118C56/HGG/SAM-DA-main/tracker/BAN/train_dataset/UAV/train.json'
__C.DATASET.UAV.FRAME_RANGE = 1
__C.DATASET.UAV.NUM_USE = -1 # 10000

__C.DATASET.VIDEOS_PER_EPOCH = 20000
__C.DATASET.VIDEOS_PER_EPOCH_B = 20000

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
__C.ALIGN.ALIGN = True

__C.ALIGN.KWARGS = CN(new_allowed=True)

# ALIGN layer type
__C.ALIGN.TYPE = "Adjust_Transformer"

# ------------------------------------------------------------------------ #
# RPN options
# ------------------------------------------------------------------------ #
__C.CAR = CN()

# RPN type
__C.CAR.TYPE = 'MultiCAR'

__C.CAR.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'SiamCARTracker'

# Scale penalty
__C.TRACK.PENALTY_K = 0.04

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.TRACK.LR = 0.4

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Context amount
__C.TRACK.CONTEXT_AMOUNT = 0.5

__C.TRACK.STRIDE = 8


__C.TRACK.SCORE_SIZE = 25

__C.TRACK.hanming = True

__C.TRACK.NUM_K = 2

__C.TRACK.NUM_N = 1

__C.TRACK.REGION_S = 0.1

__C.TRACK.REGION_L = 0.44


# ------------------------------------------------------------------------ #
# HP_SEARCH parameters [lr, pk, wi]
# ------------------------------------------------------------------------ #
__C.HP_SEARCH = CN()

__C.HP_SEARCH.UAV123 = [0.39, 0.04, 0.37] 

__C.HP_SEARCH.NAT = [0.39, 0.04, 0.37] 

__C.HP_SEARCH.NAT_L = [0.39, 0.04, 0.37] 

__C.HP_SEARCH.UAVDark70 = [0.32, 0.04, 0.36] 

__C.HP_SEARCH.NUT_L = [0.473, 0.02, 0.385]

# __C.HP_SEARCH.NUT_L = [0.39, 0.04, 0.37] 

__C.HP_SEARCH.DarkTrack2021 = [0.473, 0.06, 0.305]

__C.HP_SEARCH.UAVDark135 = [0.473, 0.06, 0.305]

__C.HP_SEARCH.WATB = [0.39, 0.04, 0.37] 

__C.HP_SEARCH.TrackingNet = [0.39, 0.04, 0.37] 
