import piqa
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

from piqa import PSNR, SSIM


class RunModel(nn.Module):
    def __init__(self, model, optimizer=None, train_loader=None, val_loader=None, test_loader=None ,mode="train", device=None):
        super(RunModel, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.mode = mode
        self.device = device
        self.best_psnr = 0.0
        self.best_ssim = 0.0
        self.device = device
        self.psnr = PSNR().cuda()
        self.ssim = SSIM().cuda()
        self.mse = nn.MSELoss()
        # 学习率调度器
        if self.optimizer is not None:
            self.linear_scheduler = LinearLR(
                optimizer, start_factor=0.001, end_factor=1.0, total_iters=100
            )
            self.multi_scheduler = MultiStepLR(
                optimizer, milestones=[1, 2], gamma=0.1
            )

        # 梯度监控
        self.gradient_stats = {
            'total_grad': [],
            'max_grad': [],
            'critical_grad': []
        }


    def _get_grad_stats(self):
        """获取梯度统计信息"""
        total_norm = 0.0
        max_norm = 0.0
        critical_grad = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_norm += grad_norm ** 2
                max_norm = max(max_norm, grad_norm)

                # 监控第一个卷积层的梯度
                if 'conv1' in name:
                    critical_grad = grad_norm

        return {
            'total_grad': total_norm ** 0.5,
            'max_grad': max_norm,
            'critical_grad': critical_grad
        }

    def _clip_gradients(self, max_norm=2.0):
        """梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=max_norm,
            norm_type=2
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch + 1}")

        for step, (img, clean_img) in enumerate(progress_bar):
            img = img.to(self.device)
            clean_img = clean_img.to(self.device)

            self.optimizer.zero_grad()

            dehaze_img = self.model(img)
            loss_mse = self.mse(dehaze_img, clean_img)
            loss_ssim = (1 - self.ssim(dehaze_img, clean_img)) ** 2

            loss = loss_mse + loss_ssim
            # loss = loss_mse
            # 反向传播与梯度缩放
            loss.backward()

            # 梯度监控与处理
            grad_stats = self._get_grad_stats()
            self._clip_gradients()

            # 参数更新
            self.optimizer.step()

            # 多步学习率衰减
            self.linear_scheduler.step()

            # 记录数据
            total_loss += loss.item()
            self.gradient_stats['total_grad'].append(grad_stats['total_grad'])
            self.gradient_stats['max_grad'].append(grad_stats['max_grad'])
            self.gradient_stats['critical_grad'].append(grad_stats['critical_grad'])

            # 更新进度条
            progress_bar.set_postfix({
                'loss_mse': loss_mse.item(),
                'loss_ssim': loss_ssim.item(),
                'lr': self.optimizer.param_groups[0]['lr'],
                'grad': f"{grad_stats['total_grad']:.3f}"
            })



        # 里程碑学习率衰减
        self.multi_scheduler.step()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def eval_model(self):
        self.model.eval()
        psnr_values = []
        ssim_values = []

        for img, clean_img in tqdm(self.val_loader, desc="Validating"):
            img = img.to(self.device)
            clean_img = clean_img.to(self.device)

            dehaze_img = self.model(img)


            # 计算图像质量指标
            psnr = self.psnr(dehaze_img, clean_img)
            ssim = self.ssim(dehaze_img, clean_img)

            psnr_values.append(psnr.item())
            ssim_values.append(ssim.item())

        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.best_ssim = avg_ssim
            torch.save(self.model.state_dict(), "IA_try.pth")

        return {
            'psnr': avg_psnr,
            'ssim': avg_ssim
        }

    @torch.no_grad()
    def test_model(self):
        self.model.eval()
        indice = 0
        for img in tqdm(self.test_loader, desc="Validating"):
            img = img.to(self.device)
            # clean_img = clean_img.to(self.device)
            dehaze_img = self.model(img)
            indice += 1

            dehaze_img = dehaze_img.squeeze(0).to('cpu')
            image_np = dehaze_img.permute(1, 2, 0).numpy()
            image_np = (image_np * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            image_pil.save(f"save_dehaze/{indice}.png")
            # print(dehaze_img.size())


    def run(self, total_epochs=12):

        if self.mode == "train":
            for epoch in range(total_epochs):
                train_loss = self.train_epoch(epoch)
                val_metrics = self.eval_model()
                print(f"\nEpoch {epoch + 1}/{total_epochs}")
                print(f"Train Loss: {train_loss:.4f}")
                print(f"Val PSNR: {val_metrics['psnr']:.2f} dB")
                print(f"Val SSIM: {val_metrics['ssim']:.4f}")
                print(f"Best PSNR: {self.best_psnr:.2f} dB")
                print(f"Best SSIM: {self.best_ssim:.2f}\n")

        else:
            self.test_model()


