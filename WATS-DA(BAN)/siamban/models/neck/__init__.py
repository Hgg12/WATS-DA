# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.models.neck.neck import AdjustLayer, AdjustAllLayer, Adjust_Transformer,EFM

NECKS = {
         'AdjustLayer': AdjustLayer,
         'AdjustAllLayer': AdjustAllLayer,
         'Adjust_Transformer': Adjust_Transformer,
         'EFM':EFM,
        }

def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
