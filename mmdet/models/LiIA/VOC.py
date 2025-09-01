import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VOCDataset_Dehaze(Dataset):
    def __init__(self, img_path=None, clean_img_path=None, tfm=None, mode="train"):
        self.img_path = img_path
        self.transform = tfm
        self.mode = mode
        self.img_files = sorted([os.path.join(self.img_path, x) for x in os.listdir(self.img_path)
                                 if x.endswith(".jpg")])

        if mode == "train":
            self.clean_img_path = clean_img_path
            self.clean_img_files = sorted([os.path.join(self.clean_img_path, x)
                                           for x in os.listdir(self.clean_img_path)
                                           if x.endswith(".jpg")])

            # assert len(self.img_files) == len(self.clean_img_files) #"训练集有雾/干净图像数量不匹配"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = self.transform(Image.open(self.img_files[idx]))

        if self.mode == "train":
            clean_img = self.transform(Image.open(self.clean_img_files[idx // 3]))
            return img, clean_img

        return img


basic_tfm = transforms.Compose([
    transforms.Resize([800, 1333]),
    transforms.ToTensor(),
    # transforms.ConvertImageDtype(torch.float),
    # transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0]),
    # transforms.Lambda(normalize_div),
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
