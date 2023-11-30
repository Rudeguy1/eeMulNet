import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import random
from utils import read_split_data, create_lr_scheduler, get_params_groups,\
    train_one_epoch, evaluate, plot_data_loader_image, plot_class_preds, test_result, test_result_addtext
from my_dataset import MyDataSet
import torchio as tio
from mobilenet3D import get_model
import sys
import time
from mobilenetv23D import get_model_mobilenetv2, get_model_mobilenetv2_addtext
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
time_start = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
strategy = 'AddText_croplabel_mobilnetv2_class2_lr5e-4batch4_resizefirst_fold0'
data_path = ''
output_path = os.path.join(data_path, strategy)

model1 = get_model_mobilenetv2_addtext(num_classes=2, sample_size=128, width_mult=1.).to(device)
Roc_best = test_result_addtext(model_path=f'{output_path}/.pth',
                        data_path='',
                       model=model1,
                        jpg_tag=f'_{strategy}')
time_end = time.time()
time_sum = time_end - time_start
print(time_sum)
print(f'the best valacc in testRoc is {Roc_best}')