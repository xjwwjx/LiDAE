import sys

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR, MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

sys.path.insert(0, "/home/xjw/MyDecetors/mmdetection")  # 路径添加

from mmdet.models.LiIA.VOC import VOCDataset_Dehaze, basic_tfm
from mmdetection.mmdet.models.LiIA.LiIA_Module import IA_Module
from mmdet.models.LiIA.RunModule import RunModel

# -------------------- 初始化部分 --------------------
# train_data_root = 'D:/xjw/Datasets/VOC_defog'
# train_data_root = '/home/xjw/Datasets/HazyDet'
#
# val_data_root = 'D:/xjw/Datasets/HazyDet'

test_data_root = '/home/xjw/Datasets/HazyDet'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 数据集
test_dataset = VOCDataset_Dehaze(
    img_path=os.path.join(test_data_root, "RDDTS/hazy_images"),
    # clean_img_path=os.path.join(test_data_root, "train/images"),
    tfm=basic_tfm,
    mode="test"
)
# train_dataset = HazyDetDataset_Dehaze(
#     img_path=os.path.join(train_data_root, "train/hazy_images"),
#     clean_img_path=os.path.join(train_data_root, "train/images"),
#     tfm=basic_tfm,
#     mode="train"
# )

# val_dataset = VOCDataset_Dehaze(
#     img_path=os.path.join(data_root, "val/hazy_images"),
#     clean_img_path=os.path.join(data_root, "val/images"),
#     tfm=basic_tfm,
#     mode="train"
# )

# val_dataset = HazyDetDataset_Dehaze(
#     img_path=os.path.join(val_data_root, "val/hazy_images"),
#     clean_img_path=os.path.join(val_data_root, "val/images"),
#     tfm=basic_tfm,
#     mode="train"
# )
# 数据加载器

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
# train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=4)

# 模型初始化

IA_Module = IA_Module().to(device)
IA_Module.load_state_dict(torch.load("IA.pth"))

# 优化器
# optimizer = SGD(
#     IA_Module.CNN_PP.parameters(),
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=0.0001
# )

# 训练器
trainer = RunModel(
    model=IA_Module,
    # optimizer=optimizer,
    # train_loader=train_loader,
    # val_loader=val_loader,
    test_loader=test_loader,
    mode='test',
    device=device
)

if __name__ == "__main__":

    trainer.run()
