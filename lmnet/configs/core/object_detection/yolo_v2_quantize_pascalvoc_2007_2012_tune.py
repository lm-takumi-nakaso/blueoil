# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from easydict import EasyDict
import tensorflow as tf
from math import log
from hyperopt import hp

from lmnet.common import Tasks
from lmnet.networks.object_detection.yolo_v2_quantize import YoloV2Quantize
from lmnet.datasets.pascalvoc_2007_2012 import Pascalvoc20072012
from lmnet.data_processor import Sequence
from lmnet.pre_processor import (
    ResizeWithGtBoxes,
    DivideBy255,
)
from lmnet.post_processor import (
    FormatYoloV2,
    ExcludeLowScoreBox,
    NMS,
)
from lmnet.data_augmentor import (
    Brightness,
    Color,
    Contrast,
    FlipLeftRight,
    Hue,
    SSDRandomCrop,
)
from lmnet.quantizations import (
    binary_channel_wise_mean_scaling_quantizer,
    linear_mid_tread_half_quantizer,
)

IS_DEBUG = False

NETWORK_CLASS = YoloV2Quantize
DATASET_CLASS = Pascalvoc20072012

IMAGE_SIZE = [416, 416]
BATCH_SIZE = 1
DATA_FORMAT = "NCHW"
TASK = Tasks.OBJECT_DETECTION
CLASSES = DATASET_CLASS.classes

MAX_EPOCHS = 100
SAVE_STEPS = 50000
TEST_STEPS = 5000
SUMMARISE_STEPS = 1000

# distributed training
IS_DISTRIBUTION = False

# for debug
# IS_DEBUG = True
# SUMMARISE_STEPS = 100
# TEST_STEPS = 1000
# SAVE_STEPS = 1000


# pretrain
IS_PRETRAIN = False
PRETRAIN_VARS = [
    'block_1/conv/kernel:0',
    'block_1/bn/beta:0',
    'block_1/bn/gamma:0',
    'block_1/bn/moving_mean:0',
    'block_1/bn/moving_variance:0',
]

PRETRAIN_DIR = "saved/core/classification/quantize_darknet_ilsvrc_2012/checkpoints"
PRETRAIN_FILE = "save.ckpt-1450000"

PRE_PROCESSOR = Sequence([
    ResizeWithGtBoxes(size=IMAGE_SIZE),
    DivideBy255()
])
anchors = [
    (1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)
]
score_threshold = 0.05
nms_iou_threshold = 0.5
nms_max_output_size = 100
POST_PROCESSOR = Sequence([
    FormatYoloV2(
        image_size=IMAGE_SIZE,
        classes=CLASSES,
        anchors=anchors,
        data_format=DATA_FORMAT,
    ),
    ExcludeLowScoreBox(threshold=score_threshold),
    NMS(iou_threshold=nms_iou_threshold, max_output_size=nms_max_output_size, classes=CLASSES,),
])

NETWORK = EasyDict()
NETWORK.OPTIMIZER_CLASS = tf.train.MomentumOptimizer
NETWORK.OPTIMIZER_KWARGS = {"momentum": 0.9}
NETWORK.LEARNING_RATE_FUNC = tf.train.piecewise_constant
# In the yolov2 paper, with a starting learning rate of 10−3, dividing it by 10 at 60 and 90 epochs.
# Train data num per epoch is 16551
# In first 5000 steps, use small learning rate for warmup.
_epoch_steps = int(16551 / BATCH_SIZE)
NETWORK.LEARNING_RATE_KWARGS = {
    "values": [1e-6, 1e-4, 1e-5, 1e-6, 1e-7],
    "boundaries": [5000, _epoch_steps * 10, _epoch_steps * 60, _epoch_steps * 90],
}
NETWORK.IMAGE_SIZE = IMAGE_SIZE
NETWORK.BATCH_SIZE = BATCH_SIZE
NETWORK.DATA_FORMAT = DATA_FORMAT
NETWORK.ANCHORS = anchors
NETWORK.OBJECT_SCALE = hp.lognormal('OBJECT_SCALE', mu=log(1), sigma=log(5))
NETWORK.NO_OBJECT_SCALE = hp.lognormal('NO_OBJECT_SCALE', mu=log(1), sigma=log(5))
NETWORK.CLASS_SCALE = hp.lognormal('CLASS_SCALE', mu=log(1), sigma=log(5))
NETWORK.COORDINATE_SCALE = hp.lognormal('COORDINATE_SCALE', mu=log(1), sigma=log(5))
NETWORK.LOSS_IOU_THRESHOLD = hp.uniform('LOSS_IOU_THRESHOLD', low=0, high=1)
NETWORK.WEIGHT_DECAY_RATE = 0.0005
NETWORK.SCORE_THRESHOLD = score_threshold
NETWORK.NMS_IOU_THRESHOLD = nms_iou_threshold
NETWORK.NMS_MAX_OUTPUT_SIZE = nms_max_output_size
NETWORK.LOSS_WARMUP_STEPS = int(12800 / BATCH_SIZE)

# quantization
NETWORK.ACTIVATION_QUANTIZER = linear_mid_tread_half_quantizer
NETWORK.ACTIVATION_QUANTIZER_KWARGS = {
    'bit': 2,
    'max_value': 2.0
}
NETWORK.WEIGHT_QUANTIZER = binary_channel_wise_mean_scaling_quantizer
NETWORK.WEIGHT_QUANTIZER_KWARGS = {}
NETWORK.QUANTIZE_FIRST_CONVOLUTION = True
NETWORK.QUANTIZE_LAST_CONVOLUTION = False

# dataset
DATASET = EasyDict()
DATASET.BATCH_SIZE = BATCH_SIZE
DATASET.DATA_FORMAT = DATA_FORMAT
DATASET.PRE_PROCESSOR = PRE_PROCESSOR
DATASET.AUGMENTOR = Sequence([
    FlipLeftRight(),
    Brightness((0.75, 1.25)),
    Color((0.75, 1.25)),
    Contrast((0.75, 1.25)),
    Hue((-10, 10)),
    SSDRandomCrop(min_crop_ratio=0.7),
])

TUNE_SPEC = {
    'num_samples': 4
}
