import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
import torch
import SimpleITK as sitk
# from volumentations import *

from scipy import ndimage
from nibabel.processing import resample_to_output
from PIL import Image
#################转换格式到np.float32关键步骤。


def read_nifti_file(filepath):
    """read and load volume"""
    #nib.load()读取文件，会将图像向左旋转90°，坐标顺序为W*H*C
    # nib ------------------------------------------------------------------
    scan = nib.load(filepath)
    img_np = scan.get_fdata(dtype=np.float32)
    # ----------------------------------------------------------------------
    # scan = sitk.ReadImage(filepath)
    # img_np = sitk.GetArrayFromImage(scan)
    return img_np


def resize_volume(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = 64
    desired_width = 128
    desired_height = 128
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate if nib read is need
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img


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
# def read_img_sitk(filepath):
#     image = sitk.ReadImage(filepath)  # <class 'SimpleITK.SimpleITK.Image'> # 支持dcm\nrrd\nii
#     image_array = sitk.GetArrayFromImage(image)  # z,y,x
#     # image_array = torch.from_numpy(image_array)
#     image_array = image_array.astype(np.float32)
#     return image_array







class MyDataSet(Dataset):
    """读取nii格式图像"""
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = read_nifti_file(self.images_path[item])
        img = normalize(img)

        # img = read_img_sitk(self.images_path[item])
        img = resize_volume(img)
        # if img.mode != 'nii':
        #     raise ValueError ("image:{} is not nii mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
        return torch.FloatTensor(img).unsqueeze(0), label

        # return torch.FloatTensor(img1).unsqueeze(0), label1

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


class MyDataSet_getpath(Dataset):
    """读取nii格式图像"""
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = read_nifti_file(self.images_path[item])
        img = normalize(img)

        # img = read_img_sitk(self.images_path[item])
        img = resize_volume(img)
        # if img.mode != 'nii':
        #     raise ValueError ("image:{} is not nii mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)
        return torch.FloatTensor(img).unsqueeze(0),label, self.images_path[item]

        # return torch.FloatTensor(img1).unsqueeze(0), label1

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        test1 = zip(*batch)
        images, labels, images_path = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        # images_paths = []
        # for i in range(len(images_path)):
        #     path = images_path[i]
        #     images_paths.append(path.split('/')[-1])

        # images_path = torch.as_tensor(images_path.split('/')[-1])
        return images, labels, images_path