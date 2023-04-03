''' 将casia-surf 数据集的三个模态分开变成三个数据集:surf_rgb,surf_depth,surf_ir,每个数据集都包含着真人和欺骗样本,并且被分为训练测试'''
'''
活体检测多模态数据caisa-surf 的dataloader
'''

# from skimage import io, transform
import cv2
from PIL import Image
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os
from lib.processing_utils import read_txt


class SURF_Single(Dataset):

    def __init__(self, txt_dir, root_dir, transform=None, modal='rgb'):
        self.related_sample_path_list = read_txt(txt_dir)
        self.root_dir = root_dir
        self.transform = transform
        self.modal = modal

    def __len__(self):
        return len(self.related_sample_path_list)

    def __getitem__(self, idx):
        related_sample_path = self.related_sample_path_list[idx]
        related_sample_path_split = related_sample_path.split(" ")

        rgb_path = os.path.join(self.root_dir, related_sample_path_split[0])
        depth_path = os.path.join(self.root_dir, related_sample_path_split[1])
        ir_path = os.path.join(self.root_dir, related_sample_path_split[2])

        binary_label = np.int64(related_sample_path_split[3])
        # print(binary_label)

        image_rgb = Image.open(rgb_path).convert('RGB')
        image_ir = Image.open(ir_path).convert('RGB')
        image_depth = Image.open(depth_path).convert('RGB')

        if self.modal == 'rgb':
            image = image_rgb
        elif self.modal == 'depth':
            image = image_depth
        elif self.modal == 'ir':
            image = image_ir
        else:
            image = None
            print("error")

        if self.transform:
            image = self.transform(image)
        return image, binary_label
