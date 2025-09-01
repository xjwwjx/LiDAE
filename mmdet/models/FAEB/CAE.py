import math
from typing import Tuple, Dict

import matplotlib
import spconv.pytorch as spconv
import torch
from functorch.dim import Tensor
from numpy.ma.core import indices
from torch import nn
import matplotlib.pyplot as plt
from torch.nn import init

from .Zero_shot_N2N import N2N_network
from mmdet.registry import MODELS
from .Context_Enhancement import CE_GN
from ..utils import unpack_gt_instances, multi_apply
import torch.nn.functional as F

from ...structures import SampleList
import numpy as np
import torch
import torch.nn as nn

@MODELS.register_module()
class Generate_Mask(nn.Module):

    def __init__(self, in_channels, hidden_channels=64, out_channels=16):
        super(Generate_Mask, self).__init__()

        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1, 0),
            nn.BatchNorm2d(hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 1, 1, 0),
            # nn.BatchNorm2d(out_channels),
            # nn.GELU(),
            # nn.ReLU()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU()
            nn.Sigmoid()
            # nn.GELU()
        )

        self.smooth = 1e-6

    def forward(self, x: Tensor, batch_data_samples=None):
        x_out = self.channel_conv(x)
        mask = self.out_conv(x_out)
        return mask

    def gen_mask(self, x: Tensor, batch_gt_instances):

        bboxes_list = [gt.bboxes for gt in batch_gt_instances]

        N, C, H, W = x.size()
        gt_mask = torch.zeros(x.size(), dtype=torch.float32).to(x.device)
        stride = 8
        max_area = 128.0
        # 遍历每个样本的bbox
        for batch_idx in range(N):
            if batch_idx >= len(bboxes_list):  # 处理样本数不一致的情况
                continue
            bbox = bboxes_list[batch_idx]
            if bbox.shape[0] == 0:  # 无bbox的样本
                continue
            scaled_bbox = torch.div(bbox, stride, rounding_mode="floor").int()
            # 遍历单个样本的所有bbox
            for obj_idx in range(scaled_bbox.shape[0]):
                x_st, y_st, x_ed, y_ed = scaled_bbox[obj_idx]
                area = (x_ed - x_st) * (y_ed - y_st)
                rate = max(0, (max_area - area) / max_area)
                value = min(1.0, max(0.2, rate ** 1))
                gt_mask[batch_idx, :, y_st:y_ed + 1, x_st:x_ed + 1] = value

        return gt_mask


#   Spatial attention block (SAB)
class SAB(nn.Module):
    def __init__(self, spatial_kernel_size=7):
        super(SAB, self).__init__()

        assert spatial_kernel_size in (3, 7, 11), 'kernel must be 3 or 7 or 11'
        self.conv_block = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel_size, padding=spatial_kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv_block(x)
        return x


# Channel attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CAB, self).__init__()

        self.in_channels = in_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.ReLU()
        self.fc1 = nn.Conv2d(self.in_channels, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        return self.sigmoid(out)


@MODELS.register_module()
class CAE(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(CAE, self).__init__()
        self.mask_network = Generate_Mask(in_channels)
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.ReLU(),
        )
        # self.mscb = MSCB(in_channels, in_channels, 1)
        self.sab = SAB()
        self.cab = CAB(in_channels, ratio=ratio)
        self.mask_tuple = None

        # self.weight = nn.Parameter(torch.tensor(0.5, device=))

    def forward(self, x: Tensor, batch_data_samples=None):
        sab_x = self.mask_network(x, batch_data_samples)
        cab_x = self.pw_conv(x)
        # mul_x = self.mscb(x)
        spatial_att = self.sab(sab_x)
        channel_att = self.cab(cab_x)
        fea_x = torch.mul(spatial_att, x)
        fea_matrix = torch.mul(channel_att, fea_x)
        x_out = fea_matrix * x + x

        return x_out

    # def loss(self, x: Tensor, batch_gt_instances) -> dict:
    #     loss = dict()
    #     loss['loss_fe'] = self.mask_network.loss(tuple(x), batch_gt_instances)
    #     return loss

    # def gen_mask(self, x: Tuple[Tensor], batch_gt_instances):
    #
    #     mask_list = []
    #     rate_list = []
    #     bboxes_list = [gt.bboxes for gt in batch_gt_instances]
    #     for i in range(len(x)):
    #         N, C, H, W = x[i].size()
    #         mask = torch.zeros((N, 1, H, W), dtype=torch.float32).to(x[i].device)
    #         stride = self.strides[i]
    #         # 遍历每个样本的bbox
    #         for batch_idx in range(N):
    #             if batch_idx >= len(bboxes_list):  # 处理样本数不一致的情况
    #                 continue
    #             bbox = bboxes_list[batch_idx]
    #             if bbox.shape[0] == 0:  # 无bbox的样本
    #                 continue
    #             scaled_bbox = torch.div(bbox, stride, rounding_mode="floor").int()
    #             # 遍历单个样本的所有bbox
    #             for obj_idx in range(scaled_bbox.shape[0]):
    #                 x_st, y_st, x_ed, y_ed = scaled_bbox[obj_idx]
    #                 mask[batch_idx, 0, y_st:y_ed + 1, x_st:x_ed + 1] = 1.0
    #         mask_list.append(mask)
    #         rate_list.append(torch.sum(mask) / mask.numel())
    #     return tuple(mask_list), tuple(rate_list)

    # def gen_mask(self, x: Tuple[Tensor], batch_gt_instances, strides, regress_ranges):
    #     mask_list = []
    #     rate_list = []
    #     bboxes_list = [gt.bboxes for gt in batch_gt_instances]
    #
    #     for i, feat in enumerate(x):
    #         N, C, H, W = feat.size()
    #         mask = torch.zeros((N, 1, H, W), dtype=torch.float32, device=feat.device)
    #
    #         for batch_idx in range(N):
    #             if batch_idx >= len(bboxes_list) or bboxes_list[batch_idx].shape[0] == 0:
    #                 continue
    #
    #             # 转换bbox到特征图坐标（假设bbox格式为xyxy）
    #             bbox = bboxes_list[batch_idx]
    #             # scale_bbox = torch.zeros_like(bbox, dtype=torch.float32)
    #             # 计算所有中心点
    #             y_center = (bbox[:, 1] + bbox[:, 3]) / 2
    #             x_center = (bbox[:, 0] + bbox[:, 2]) / 2
    #             centers = torch.stack([y_center, x_center], dim=1)  # (num_objs, 2)
    #
    #             # 生成坐标网格
    #             y_coords = torch.arange(start=0, end=H * strides[i], step=strides[i], dtype=torch.float32, device=feat.device)
    #             x_coords = torch.arange(start=0, end=W * strides[i], step=strides[i], dtype=torch.float32, device=feat.device)
    #             grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    #
    #             # 计算每个点到所有中心点的最小距离
    #             dists = torch.cdist(
    #                 torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).unsqueeze(0),
    #                 centers.unsqueeze(0)
    #             ).squeeze(0)
    #
    #             min_dists = dists.min(dim=1)[0]
    #             mask[batch_idx, 0] = min_dists.view(H, W)
    #
    #         min_mask = (mask >= regress_ranges[i][0])
    #         max_mask = (mask <= regress_ranges[i][1])
    #
    #         mask = (min_mask & max_mask).float()
    #         mask_rate = torch.sum(mask) / mask.numel()
    #         mask_list.append(mask)
    #         rate_list.append(mask_rate)
    #
    #     return tuple(mask_list), tuple(rate_list)
