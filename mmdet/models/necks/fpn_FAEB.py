from functools import partial
from typing import List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor
from mmdetection.mmdet.models.FAEB.CAE import CAE
from mmdetection.mmdet.models.FAEB.UTCB import UTCB

from mmdet.registry import MODELS
from mmdet.utils import MultiConfig


@MODELS.register_module()
class Custom_FPN(BaseModule):

    def __init__(
            self,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            num_fe: int,
            start_level: int = 0,
            upsample_mode: str = 'nearest',
            init_cfg: MultiConfig = dict(
                type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels

        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.num_fe = num_fe

        self.upsample_mode = upsample_mode
        self.backbone_end_level = self.num_ins

        assert num_outs >= self.num_ins - start_level
        self.start_level = start_level

        self.down_sample = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        self.lateral_convs = nn.ModuleList()
        self.up_eucbs = nn.ModuleList()
        self.fus_convs = nn.ModuleList()
        self.fe_layers = nn.ModuleList()

        # self.up = nn.Upsample(scale_factor=2)

        for i in range(self.start_level, self.backbone_end_level):

            l_conv = nn.Conv2d(in_channels[i], out_channels, 1, 1, 0)

            up_eucb = UTCB(out_channels, out_channels)

            fus_conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

            self.lateral_convs.append(l_conv)
            self.up_eucbs.append(up_eucb)
            self.fus_convs.append(fus_conv)

        for i in range(self.num_fe):
            fe_layer = CAE(out_channels, ratio=16)
            self.fe_layers.append(fe_layer)

    def forward(self, inputs: Tuple[Tensor]) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals

        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + self.up_eucbs[i - 1](laterals[i])
            # laterals[i - 1] = laterals[i - 1] + self.up(laterals[i])

        outs = [
            self.fus_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        for i in range(self.num_fe):
            outs[i] = self.fe_layers[i](outs[i])

        extra_p6 = self.down_sample(laterals[used_backbone_levels - 1])

        outs.append(extra_p6)

        return tuple(outs)
