import torch
from torch import nn

from mmdet.registry import MODELS
from .LiIA_Module import LiIA_Module


@MODELS.register_module(name='IA')
@MODELS.register_module()
class LiIA(nn.Module):
    """Lightweight Image Adaptive (LiIA) dehazing/enhancement wrapper.

    Registered as both 'IA' (legacy name used by the released configs and
    checkpoints) and 'LiIA'. The inner module attribute is named
    ``IA_Module`` so that released checkpoints load with strict=True.
    """

    def __init__(self, init_cfg=None, mean=None, std=None):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.IA_Module = LiIA_Module()

        if init_cfg is not None:
            checkpoint = init_cfg.get('checkpoint') if isinstance(
                init_cfg, dict) else getattr(init_cfg, 'checkpoint', None)
            if checkpoint is not None:
                state = torch.load(checkpoint, map_location='cpu')
                self.IA_Module.load_state_dict(state)

    def forward(self, img):

        mean = self.mean.to(img.device)
        std = self.std.to(img.device)

        max = torch.tensor(255.0, dtype=torch.float32, device=img.device)
        img = img * std + mean
        img = img / max

        img = self.IA_Module(img)

        img = img * max
        img = (img - mean) / std

        return img
