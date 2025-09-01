import torch
from torch import nn
from torch.nn import init

from mmdet.registry import MODELS


@MODELS.register_module()
class CE_GN(nn.Module):
    def __init__(self, in_channels=256):
        super(CE_GN, self).__init__()

        self.w = nn.Parameter(torch.ones((1, in_channels, 1, 1)))
        # 偏置初始化为0
        self.b = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

    def forward(self, feature,  context):
        # 计算每个通道的均值和标准差

        mean = context.mean(dim=1, keepdim=True)
        std = context.std(dim=1, keepdim=True) + 1e-5


        # 按通道归一化
        feature = self.w * (feature - mean) / std + self.b
        # feature = self.w * (feature - mean) / std
        feature += context

        return feature
