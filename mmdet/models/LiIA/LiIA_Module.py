import torch
import torch.nn.functional as F
from torch import nn

from .cfg import cfg
from ..FAEB import WTConv2d


class WCPG(nn.Module):
    def __init__(self, config=None):
        super(WCPG, self).__init__()
        self.cfg = config
        self.trainable = self.cfg.trainable
        self.channels = self.cfg.channels
        self.output_dim = self.cfg.num_filter_parameters

        # Define the layers
        self.conv0 = self._conv_block(3, 2 * self.channels, downsample=True)
        self.conv1 = self._wtconv_block(2 * self.channels, 2 * self.channels, downsample=True)
        self.conv2 = self._wtconv_block(2 * self.channels, 2 * self.channels, downsample=True)

        self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=4)
        self.max_pool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.channel_conv = nn.Conv2d(4 * self.channels, self.channels, kernel_size=1, stride=1, padding=0)


        self.fc1 = nn.Linear(1536, 64)
        self.fc2 = nn.Linear(64, self.output_dim)

    def _conv_block(self, in_channels, out_channels, kernel_size=3, downsample=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2 if downsample else 1,
                            padding=kernel_size // 2),
                  nn.BatchNorm2d(out_channels),
                  nn.LeakyReLU(negative_slope=0.1)]

        return nn.Sequential(*layers)

    def _wtconv_block(self, in_channels, out_channels, kernel_size=3, wt_level=1, downsample=False):
        layers = [
            WTConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2 if downsample else 1,
                     wt_levels=wt_level),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        ]

        return nn.Sequential(*layers)

    def forward(self, x):
        # x.shape: 3*256*384

        x = self.conv0(x)  # x.shape: 16*128*192
        x = self.conv1(x)  # x.shape: 32*64*96
        x = self.conv2(x)  # x.shape: 32*32*48

        x = torch.cat([self.avg_pool(x), self.max_pool(x)], dim=1)  # x.shape: 16*8*12
        x = self.channel_conv(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor to feed into FC layers
        features = F.leaky_relu(self.fc1(x), negative_slope=0.1)
        filter_features = self.fc2(features)
        return filter_features



class LiIA_Module(nn.Module):

    def __init__(self, config=cfg):
        super(LiIA_Module, self).__init__()
        self.cfg = config
        self.WCPG = WCPG(config=self.cfg)
        self.FG = [x(self.cfg) for x in self.cfg.filters]
        self.mse = nn.MSELoss()

    def forward(self, feature):
        resized_feature = F.interpolate(feature, size=(256, 384), mode='bilinear', align_corners=False)

        params = self.WCPG(resized_feature)

        for per_filter in self.FG:
            feature, _ = per_filter.apply(feature, params)

        # return feature
        return torch.clamp(feature, 0.0, 1.0)
