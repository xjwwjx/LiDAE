## https://arxiv.org/abs/2303.11253

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.registry import MODELS


@MODELS.register_module()
class N2N_network(nn.Module):

    def __init__(self, in_channels, chan_embed=48):
        super(N2N_network, self).__init__()
        self.BatchNorm1 = nn.BatchNorm2d(in_channels)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(in_channels, chan_embed, 3, padding=1)
        self.BatchNorm2 = nn.BatchNorm2d(chan_embed)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.BatchNorm3 = nn.BatchNorm2d(chan_embed)
        self.conv3 = nn.Conv2d(chan_embed, in_channels, 1)
        self.mse = nn.MSELoss()

    def pair_downsampler(self, noisy_img):
        # img has shape B C H W
        c = noisy_img.shape[1]
        filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(noisy_img.device)
        filter1 = filter1.repeat(c, 1, 1, 1)
        filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(noisy_img.device)
        filter2 = filter2.repeat(c, 1, 1, 1)
        img1 = F.conv2d(noisy_img, filter1, stride=2, groups=c)
        img2 = F.conv2d(noisy_img, filter2, stride=2, groups=c)
        return img1, img2

    def dis_noise(self, noisy_img):
        x = noisy_img
        x = self.BatchNorm1(x)
        x = self.act(self.conv1(x))
        x = self.BatchNorm2(x)
        x = self.act(self.conv2(x))
        x = self.BatchNorm3(x)
        x = self.conv3(x)
        return x

    def loss(self, noisy_img):
        noisy1, noisy2 = self.pair_downsampler(noisy_img)

        pred1 = noisy1 - self.dis_noise(noisy1)
        pred2 = noisy2 - self.dis_noise(noisy2)

        loss_res = 1 / 2 * (self.mse(noisy1, pred2) + self.mse(noisy2, pred1))

        noisy_denoised = noisy_img - self.dis_noise(noisy_img)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)

        loss_cons = 1 / 2 * (self.mse(pred1, denoised1) + self.mse(pred2, denoised2))

        loss = loss_res + loss_cons
        return loss / (noisy_img.shape[0] * 20)

    def forward(self, noisy_img):
        noise = self.dis_noise(noisy_img)
        clean_img = noisy_img - noise
        # clean_img = torch.clamp(noisy_img - noise, 0, 1)
        return clean_img
