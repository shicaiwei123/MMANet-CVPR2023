'''
活体检测多模态数据caisa-surf 的dataloader
'''

# from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os
from lib.processing_utils import read_txt, get_file_list
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as tt


class RandomHorizontalFlip_multi(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label']

        if random.random() < self.p:
            new_image_x = cv2.flip(image_x, 1)
            new_image_ir = cv2.flip(image_ir, 1)
            new_image_depth = cv2.flip(image_depth, 1)
            return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                    'binary_label': binary_label}
        else:
            return sample


class Resize_multi(object):

    def __init__(self, size):
        '''
        元组size,如(112,112)
        :param size:
        '''
        self.size = size

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label']

        new_image_x = cv2.resize(image_x, self.size)
        new_image_ir = cv2.resize(image_ir, self.size)
        new_image_depth = cv2.resize(image_depth, self.size)

        return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                'binary_label': binary_label}


class RondomRotion_multi(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label']

        (h, w) = image_x.shape[:2]
        (cx, cy) = (w / 2, h / 2)

        # 设置旋转矩阵
        angle = random.randint(-self.angle, self.angle)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        cos = np.abs(M[0, 0]) * 0.8
        sin = np.abs(M[0, 1]) * 0.8

        # 计算图像旋转后的新边界
        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
        M[0, 2] += (nw / 2) - cx
        M[1, 2] += (nh / 2) - cy

        new_image_x = cv2.warpAffine(image_x, M, (nw, nh))
        new_image_ir = cv2.warpAffine(image_ir, M, (nw, nh))
        new_image_depth = cv2.warpAffine(image_depth, M, (nw, nh))

        return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                'binary_label': binary_label}


class RondomCrop_multi(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label']

        h, w = image_x.shape[:2]

        y = np.random.randint(0, h - self.size)
        x = np.random.randint(0, w - self.size)

        new_image_x = image_x[y:y + self.size, x:x + self.size, :]
        new_image_ir = image_ir[y:y + self.size, x:x + self.size, :]
        new_image_depth = image_depth[y:y + self.size, x:x + self.size, :]

        return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                'binary_label': binary_label}


class Cutout_multi(object):
    '''
    作用在to tensor 之后
    '''

    def __init__(self, length=30):
        self.length = length

    def __call__(self, sample):
        img, image_ir, image_depth, binary_label = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label']
        h, w = img.shape[1], img.shape[2]  # Tensor [1][2],  nparray [0][1]
        length_new = np.random.randint(1, self.length)
        y = np.random.randint(h - length_new)
        x = np.random.randint(w - length_new)

        img[y:y + length_new, x:x + length_new] = 0
        image_ir[y:y + length_new, x:x + length_new] = 0
        image_depth[y:y + length_new, x:x + length_new] = 0

        return {'image_x': img, 'image_ir': image_ir, 'image_depth': image_depth, 'binary_label': binary_label}


class Normaliztion_multi(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __init__(self):
        self.a = 1

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label']

        new_image_x = (image_x - 127.5) / 128  # [-1,1]
        new_image_ir = (image_ir - 127.5) / 128  # [-1,1]
        new_image_depth = (image_depth - 127.5) / 128  # [-1,1]

        return {'image_x': new_image_x, 'image_ir': new_image_ir, 'image_depth': new_image_depth,
                'binary_label': binary_label}


class ToTensor_multi(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __init__(self):
        self.a = 1

    def __call__(self, sample):
        image_x, image_ir, image_depth, binary_label = sample['image_x'], sample['image_ir'], sample[
            'image_depth'], sample['binary_label']

        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x.transpose((2, 0, 1))
        image_x = np.array(image_x)

        image_ir = image_ir.transpose((2, 0, 1))
        image_ir = np.array(image_ir)

        image_depth = image_depth.transpose((2, 0, 1))
        image_depth = np.array(image_depth)

        binary_label = np.array(binary_label)

        return {'image_x': torch.from_numpy(image_x.astype(np.float32)).float(),
                'image_ir': torch.from_numpy(image_ir.astype(np.float32)).float(),
                'image_depth': torch.from_numpy(image_depth.astype(np.float32)).float(),
                'binary_label': torch.from_numpy(binary_label)}


class SURF(Dataset):

    def __init__(self, txt_dir, root_dir, miss_modal, fill=0, transform=None, times=1):
        self.related_sample_path_list = read_txt(txt_dir)

        # img_index = []
        #
        # for idx in range(len(self.related_sample_path_list)):
        #     related_sample_path = self.related_sample_path_list[idx]
        #     related_sample_path_split = related_sample_path.split(" ")
        #     rgb_related_sample_path_split = related_sample_path_split[0]
        #     img_index.append(rgb_related_sample_path_split.split('/')[-1])
        #
        # # 利用img_index对self.related_sample_path_list排序,使得能够按照顺序读取txt
        # img_index, self.related_sample_path_list = (list(t) for t in
        #                                             zip(*sorted(zip(img_index, self.related_sample_path_list))))

        self.root_dir = root_dir
        self.transform = transform
        self.miss_modal = miss_modal
        self.fill = fill
        self.times = times
        self.real_len = len(self.related_sample_path_list)

    def __len__(self):
        return len(self.related_sample_path_list) * self.times
        # return 5

    def __getitem__(self, idx):
        if idx >= self.real_len:
            idx = idx % self.real_len
        related_sample_path = self.related_sample_path_list[idx]
        # print(related_sample_path)
        related_sample_path_split = related_sample_path.split(" ")

        rgb_path = os.path.join(self.root_dir, related_sample_path_split[0])
        depth_path = os.path.join(self.root_dir, related_sample_path_split[1])
        ir_path = os.path.join(self.root_dir, related_sample_path_split[2])

        binary_label = np.int64(related_sample_path_split[3])

        # print(rgb_path)
        # print(ir_path)
        # print(depth_path)
        image_rgb = cv2.imread(rgb_path)
        image_ir = cv2.imread(ir_path)
        image_depth = cv2.imread(depth_path)

        # 模态缺失调整
        image_rgb, image_ir, image_depth = self.modal_adjust(image_rgb, image_ir, image_depth, fill=self.fill)

        sample = {'image_x': image_rgb, 'image_depth': image_depth, 'image_ir': image_ir, 'binary_label': binary_label}

        if self.transform:
            sample = self.transform(sample)

        # print(sample)
        return sample

    def modal_adjust(self, img_rgb, img_ir, img_depth, fill=0):
        if self.miss_modal == 1:
            img_size = img_rgb.shape
            img_rgb = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_rgb = np.uint8(img_rgb)

        elif self.miss_modal == 2:
            img_size = img_ir.shape
            img_ir = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_ir = np.uint8(img_ir)

        elif self.miss_modal == 3:
            img_size = img_depth.shape
            img_depth = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_depth = np.uint8(img_depth)

        elif self.miss_modal == 4:
            img_size = img_ir.shape
            img_ir = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_ir = np.uint8(img_ir)

            img_size = img_depth.shape
            img_depth = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_depth = np.uint8(img_depth)

        elif self.miss_modal == 5:
            img_size = img_rgb.shape
            img_rgb = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_rgb = np.uint8(img_rgb)

            img_size = img_depth.shape
            img_depth = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_depth = np.uint8(img_depth)

        elif self.miss_modal == 6:
            img_size = img_ir.shape
            img_ir = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_ir = np.uint8(img_ir)

            img_size = img_ir.shape
            img_rgb = np.ones((img_size[0], img_size[1], img_size[2])) * fill
            img_rgb = np.uint8(img_rgb)

        else:
            img_rgb = img_rgb
            img_ir = img_ir
            img_depth = img_depth
        return img_rgb, img_ir, img_depth


class SURF_generate(Dataset):
    def __init__(self, rgb_dir, depth_dir, ir_dir, transform=None):
        '''

        :param rgb_dir:  dataroot for rgb modal
        :param depth_dir:  dataroot for depth modal
        :param ir_dir: dataroot for ir modal
        '''
        self.rgb_path_list = get_file_list(rgb_dir)
        self.rgb_path_list.sort()
        self.depth_path_list = get_file_list(depth_dir)
        self.depth_path_list.sort()
        self.ir_path_list = get_file_list(ir_dir)
        self.ir_path_list.sort()

        self.transform = transform

    def __getitem__(self, idx):
        rgb_img = cv2.imread(self.rgb_path_list[idx])
        ir_img = cv2.imread(self.ir_path_list[idx])
        depth_img = cv2.imread(self.depth_path_list[idx])

        rgb_path_split = self.rgb_path_list[idx].split('/')
        if rgb_path_split[-3] == 'real_park':
            binary_label = 1
        elif rgb_path_split[-3] == 'fake_park':
            binary_label = 0
        else:
            print("label error")
            binary_label = 2

        sample = {'image_x': rgb_img, 'image_ir': depth_img, 'image_depth': ir_img, 'binary_label': binary_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.rgb_path_list)
        # return 100
