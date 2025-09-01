from easydict import EasyDict as eDict
from .FG import *

cfg = eDict()
cfg.trainable = True
cfg.channels = 16
# cfg.out_channels = 8


cfg.filters = [
    DefogFilter, ImprovedWhiteBalanceFilter,  GammaFilter,
    ToneFilter, ContrastFilter, UsmFilter
]

cfg.num_filter_parameters = 16

cfg.defog_begin_param = 0
cfg.wb_begin_param = 1
cfg.gamma_begin_param = 4
cfg.tone_begin_param = 5
cfg.contrast_begin_param = 13
cfg.usm_begin_param = 14
cfg.mask_token = 15

cfg.curve_steps = 8
cfg.gamma_range = 3
cfg.exposure_range = 3.5
cfg.wb_range = 1.1
cfg.color_curve_range = (0.90, 1.10)
cfg.lab_curve_range = (0.90, 1.10)
cfg.tone_curve_range = (0.5, 2)
cfg.defog_range = (0.1, 1.0)
cfg.usm_range = (0.0, 5)

