from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mmdet.registry import MODELS
from mmcv.cnn import ConvModule

from mmdet.utils import InstanceList


class DCKModule(nn.Module):
    def __init__(self, channels,
                 num_layers=1,
                 kernel_size=7,
                 group_channels=16,
                 reduction_ratio=4):
        super(DCKModule, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        self.num_layers = num_layers
        self.group_channels = group_channels
        self.reduction_ratio = reduction_ratio
        self.groups = self.channels // self.group_channels

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for _ in range(num_layers):
            self.convs1.append(ConvModule(
                in_channels=channels,
                out_channels=channels // reduction_ratio,
                kernel_size=1,
                conv_cfg=None,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='ReLU')))
            self.convs2.append(nn.Conv2d(
                in_channels=channels // reduction_ratio,
                out_channels=self.groups * self.kernel_size * self.kernel_size,
                kernel_size=1,
                stride=1,
                bias=False))

        self.padding = (kernel_size - 1) // 2

    def forward(self, feature_map, guide_map):
        b, c, h, w = feature_map.size()
        gc = self.group_channels
        g = self.groups
        n = self.kernel_size * self.kernel_size
        for i in range(self.num_layers):
            # 生成动态卷积核
            dynamic_filters = self.convs2[i](self.convs1[i](guide_map))
            # dynamic_filters 形状: (b, g * n, h, w)
            dynamic_filters = dynamic_filters.view(b, g, n, h, w)
            dynamic_filters = dynamic_filters.permute(0, 3, 4, 1, 2)  # 形状: (b, h, w, g, n)

            # 提取输入特征图的拼块
            input_patches = F.unfold(feature_map, kernel_size=self.kernel_size,
                                     padding=self.padding)  # (b, c * n, h * w)
            input_patches = input_patches.view(b, c, n, h, w)  # (b, c, n, h, w)
            input_patches = input_patches.view(b, g, gc, n, h, w)  # (b, g, gc, n, h, w)
            input_patches = input_patches.permute(0, 4, 5, 1, 3, 2)  # 形状: (b, h, w, g, n, gc)

            # 计算输出
            out = torch.einsum('bhwgnc,bhwgn->bhwgc', input_patches, dynamic_filters)
            out = out.permute(0, 3, 4, 1, 2).contiguous()  # 形状: (b, g, gc, h, w)
            out = out.view(b, c, h, w)

            # 残差连接
            feature_map = out + feature_map
        return feature_map


@MODELS.register_module()
class DepthCue(nn.Module):
    def __init__(self, in_channels, seg_out_channels=1):
        super(DepthCue, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = self.in_channels // 8
        self.seg_out_channels = seg_out_channels
        self.Filter = DCKModule(self.in_channels)
        self.Sirloss = SIRLOSS(smooth=True, log=True)
        self.network = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(num_groups=8, num_channels=self.in_channels),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(num_groups=8, num_channels=self.in_channels),
            nn.ReLU(),
        )

        self.conv_seg = nn.Sequential(
            nn.Conv2d(self.in_channels, 1, kernel_size=3, stride=1, padding=1),
            # nn.GroupNorm(num_groups=1, num_channels=1),
            nn.ReLU()
        )

    def forward(self, x):
        seg_feat = self.network(x)
        return self.Filter(x, seg_feat)

    def loss(self, x: tuple[Tensor], batch_gt_sem_seg):
        seg_feat_list = []

        for per_x in x:
            per_x = self.network(per_x)
            seg_score = self.conv_seg(per_x)
            seg_feat_list.append(seg_score)

        seg_targets = self.get_seg_targets(seg_feat_list, batch_gt_sem_seg)

        loss = dict()
        indice = -1

        for seg_feat in seg_feat_list:
            indice += 1
            seg_scores = seg_feat.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            filter_loss = self.Sirloss(seg_scores, seg_targets[indice])

            if loss.get('loss_filter') is None:
                loss['loss_filter'] = filter_loss
            else:
                loss['loss_filter'] += filter_loss

        loss['loss_filter'] = 0.1 * loss['loss_filter']

        return loss

    def get_seg_targets(
            self,
            seg_scores: List[Tensor],
            batch_gt_sem_seg: InstanceList
    ) -> List[Tensor]:
        """
        Prepare segmentation targets.

        Args:
            seg_scores (List[Tensor]): List of segmentation scores of different
                levels in shape (batch_size, num_classes, h, w)
            batch_gt_instances (InstanceList): Ground truth instances.

        Returns:
            List[Tensor]: Segmentation targets of different levels.
        """
        lvls = len(seg_scores)
        batch_size = len(batch_gt_sem_seg)
        assert batch_size == seg_scores[0].size(0)
        seg_targets = []

        for lvl in range(lvls):
            _, _, h, w = seg_scores[lvl].shape
            lvl_seg_target = []
            for gt_sem_seg in batch_gt_sem_seg:
                lvl_seg_target.append(gt_sem_seg.sem_seg.data)
            lvl_seg_target = torch.stack(lvl_seg_target, dim=0)
            lvl_seg_target = F.interpolate(lvl_seg_target,
                                           size=(h, w), mode='nearest')
            seg_targets.append(lvl_seg_target)

        # flatten seg_targets
        flatten_seg_targets = [
            seg_target.permute(0, 2, 3, 1).reshape(-1, self.seg_out_channels)
            for seg_target in seg_targets
        ]

        return flatten_seg_targets


class SIRLOSS(nn.Module):
    """Scale-Invariant Logarithmic (SiLog) Loss.

    Args:
        bsize (int): Batch size.
        eduction (str, optional): The method to reduce the loss.
                Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss.
        avg_non_ignore (bool, optional): If True, only consider non-ignored elements for averaging. Defaults to False.
    """

    def __init__(self,
                 ignore_index=255,
                 loss_weight: float = 1.0,
                 smooth: bool = False,
                 epsilon: float = 0.1,
                 log: bool = False) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.smooth = smooth
        self.epsilon = epsilon
        self.log = log

    def forward(self,
                pred: Tensor,
                label: Tensor,
                ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            label (Tensor): The target tensor.
            mask (Tensor): The mask tensor.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.

        Returns:
            Tensor: Calculated loss.
        """
        # The default value of ignore_index is the same as in `F.cross_entropy`
        ignore_index = -100 if self.ignore_index is None else self.ignore_index

        # Mask out ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()

        if self.log:
            # Ensure pred_valid and label_valid are positive before taking log
            pred = torch.clamp(pred, min=1e-2)
            label = torch.clamp(label, min=1e-2)
            pred = torch.log(pred)
            label = torch.log(label)
            # print(pred.min(), pred.max())
        if self.smooth:
            label = (1 - self.epsilon) * label + self.epsilon * pred

        pred_valid = pred * valid_mask
        label_valid = label * valid_mask

        diff = (pred_valid - label_valid)

        nvalid_pix = torch.sum(valid_mask)

        depth_cost = (torch.sum(nvalid_pix * torch.sum(diff ** 2))
                      - 0.5 * torch.sum(torch.sum(diff) ** 2)) \
                     / torch.maximum(torch.sum(nvalid_pix ** 2), torch.tensor(1.0, device=label.device))

        return self.loss_weight * depth_cost
