import json
from torch.utils.data.dataset import Dataset
import os
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from volumentations import *
import torchio as tio
from torchvision import transforms
from scipy import ndimage
import torch

def read_nifti_file(filepath):
    """read and load volume"""
    #nib.load()读取文件，会将图像向左旋转90°，坐标顺序为W*H*C
    # nib ------------------------------------------------------------------
    # scan = nib.load(filepath)
    # img_np = scan.get_fdata(dtype=np.float32)
    # ----------------------------------------------------------------------
    scan = sitk.ReadImage(filepath)
    img_np = sitk.GetArrayFromImage(scan)
    return img_np

def normalize(volume):
    """Normalize the volume"""
    # min = -1000
    # max = 400
    volume_min = np.min(volume)
    volume_max = np.max(volume)
    volume = (volume - volume_min) / (volume_max - volume_min)
    # volume_mean = np.mean(volume)
    # volume_sigma = np.std(volume)
    # volume = (volume-volume_mean) / volume_sigma
    volume = volume.astype("float32")
    return volume


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[0]
    current_width = img.shape[1]
    current_height = img.shape[2]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate if nib read is need
    # img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (depth_factor,  width_factor, height_factor), order=1)
    return img

class classification_dataset(Dataset):
    def __init__(self, dir_path, classification_json, transform):
        self.dir = dir_path
        self.transform = transform
        self.imgs = []
        self.labels = []

        json_path = os.path.join(dir_path, classification_json)
        with open(json_path) as j:
            datas = json.load(j)

        for data, label in tqdm(datas.items()):
            img_path = f'{dir_path}{data}'
            self.imgs.append(img_path)
            self.labels.append(label)

    def __getitem__(self, item):
        img = read_nifti_file(self.imgs[item])
        label = self.labels[item]
        # aug img ----------------------------------------------------------------
        img = resize_volume(img)
        data_structure = {'image': img}
        if self.transform is not None:
            transform_img = self.transform(**data_structure)
            img = transform_img['image']
        img = normalize(img)
        img = transforms.ToTensor(img)
        return torch.FloatTensor(img).unsqueeze(0), label
        # ----------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels




# ----------------------------------------------------------------------------test
# train_transoform = Compose([
#     Flip(0, p=0.5), Flip(1, p=0.5), Flip(2, p=0.5),
#     Rotate((-15, 15), (-5, 5), (-45, 45), p=0.5),
#     RandomRotate90((1, 2), p=0.5),
#     GaussianNoise(var_limit=(0, 5), p=0.5)
# ], p=1)
# test1 = classification_dataset(dir_path='/home/zhanggf/code/pythonProject_exercise/data/allAug_pad_gray_crop_compresion_12812864', classification_json='0_train.json', transform=train_transoform)
# test1.__getitem__(1)
# ----------------------------------------------------------------------------------