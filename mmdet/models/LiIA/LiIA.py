import torch
from torch import nn

from mmdet.registry import MODELS
from .LiIA_Module import LiIA_Module


@MODELS.register_module()
class LiIA(nn.Module):
    def __init__(self, init_cfg=None, mean=None, std=None):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.init_cfg = init_cfg
        self.LiIA_Module = LiIA_Module()

        if self.init_cfg is not None:
            self.IA_Module.load_state_dict(torch.load(self.init_cfg.checkpoint))
            # self.IA_Module.eval()
            # for param in self.IA_Module.parameters():
            #     param.requires_grad = True

    def forward(self, img):

        mean = self.mean.to(img.device)
        std = self.std.to(img.device)

        max = torch.tensor(255.0, dtype=torch.float32, device=img.device)
        img = img * std + mean
        img = img / max

        img = self.LiIA_Module(img)

        img = img * max
        img = (img - mean) / std

        return img
