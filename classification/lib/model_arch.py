import torch.nn as nn
import torch
import torchvision.models as tm
import torch.nn.functional as F
import numpy as np
import random


class ROI_Pooling(nn.Module):
    '''
    处理单个feature map的 roi 图像信息
    '''

    def __init__(self):
        super().__init__()
        self.avgpool_patch = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool_patch = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, feature_map, cluster_center, spatial_ratio):
        feature_list = []
        cluster_center_mean = torch.mean(cluster_center, dim=0)
        cluster_center_normal = cluster_center_mean / spatial_ratio
        cluster_center_int = torch.floor(cluster_center_normal)
        cluster_center_float = cluster_center_normal - cluster_center_int
        cluster_center_offset = torch.round(cluster_center_float)
        cluster_center_offset = cluster_center_offset * 2 - 1  # 转到[-1,1]
        cluster_center_int = cluster_center_int + 1  # 转到[1,5]
        cluster_center_int = cluster_center_int + cluster_center_offset

        padding = (1, 1, 1, 1)
        # feature_map = F.pad(feature_map, padding, 'constant', 1)

        # for index in range(cluster_center_mean.shape[0]):
        #     coordinate_single = cluster_center_int[index]
        #     coordinate_single=coordinate_single.long()
        #     # x2 是因为python 索引的问题,从0开始,[0:1] 只索引一个
        #
        #     patch = feature_map[:, :,
        #                         coordinate_single[0]:coordinate_single[0] + 2,
        #                         coordinate_single[1]:coordinate_single[1] + 2]
        #
        patch_avg = self.avgpool_patch(feature_map)
        patch_max = self.maxpool_patch(feature_map)
        patch_feature = patch_avg
        patch_flatten = torch.flatten(patch_feature, 1)
        feature_list.append(patch_flatten)

        return feature_list


class SpatialAttention(nn.Module):
    '''
    空间注意力模块
    '''

    def __init__(self, kernel_size=1):
        super(SpatialAttention, self).__init__()

        padding = 0

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # self.avg = nn.AdaptiveAvgPool2d((3, 3))

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def modality_drop(x_rgb, x_depth,x_ir, p, args):
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    index_list = [x for x in range(7)]

    if p == [0, 0, 0]:
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        p = np.array(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    else:
        p = p
        # print(p)
        p = [p * x_rgb.shape[0]]
        # print(p)
        p = np.array(p).reshape(x_rgb.shape[0], 3)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

        # print(p[:, 0], p[:, 1], p[:, 2])
    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_depth = x_depth * p[:, 1]
    x_ir = x_ir * p[:, 2]

    return x_rgb, x_depth,x_ir, p



def unbalance_modality_drop(x_rgb, x_depth,x_ir, p, args):
    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    index_list = [x for x in range(7)]
    prob = np.array((1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7, 1 / 7))
    # print(args.epoch)
    mode_num = 7
    hard_mode_index = [0, 2, 4]
    mode_average = x_rgb.shape[0] // mode_num
    batch_left = x_rgb.shape[0] % mode_num
    mode_left = 2
    if p == [0, 0, 0]:
        p = []
        # prob = np.array([3 / 12, 1 / 12, 3 / 12, 1 / 12, 2 / 12, 1 / 12, 1 / 12])
        # for i in range(x_rgb.shape[0]):
        #     index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
        #     p.append(modality_combination[index])
        #     # if 'model_arch_index' in args.writer_dicts.keys():
        #     #     args.writer_dicts['model_arch_index'].write(str(index) + " ")
        #
        # p = np.array(p)
        # p = torch.from_numpy(p)
        # p = torch.unsqueeze(p, 2)
        # p = torch.unsqueeze(p, 3)
        # p = torch.unsqueeze(p, 4)

        # if args.epoch < 15:
        #     for i in range(mode_num):
        #         p = p + modality_combination[i] * mode_average
        #     for i in range(batch_left):
        #         p = p + modality_combination[i]
        # else:
        #     increase_num =  args.epoch - 15
        #     if increase_num > 7:
        #         increase_num = 7
        #
        #     # print(increase_num)
        #     for i in hard_mode_index:
        #         p = p + modality_combination[i] * (mode_average + increase_num)
        #
        #     decrease_num = args.epoch - 15
        #     if decrease_num > 7:
        #         decrease_num = 7
        #
        #     # print(decrease_num)
        #     for i in [3,5,6]:
        #         p = p + modality_combination[i] * (mode_average - decrease_num)
        #     p=p + modality_combination[1] * mode_average
        #     for i in range(batch_left):
        #         p = p + modality_combination[i]

        # p = p + modality_combination[2] * 17
        # for i in [0, 4]:
        #     p = p + modality_combination[i] * 11
        # for i in [1, 3, 5]:
        #     p = p + modality_combination[i] * 7
        # p = p + modality_combination[6] * 4



        p = []
        prob = np.array((1 / 4, 1 / 4, 1 / 4, 0, 0, 0, 1/4))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])



        p = np.array(p)
        p = p.reshape((64, 3))
        np.random.shuffle(p)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)



    else:
        p = p
        p = [p * x_rgb.shape[0]]
        p = np.array(p).reshape(x_rgb.shape[0], 3)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

        # print(p[:, 0], p[:, 1], p[:, 2])
    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_depth = x_depth * p[:, 1]
    x_ir = x_ir * p[:, 2]

    return x_rgb, x_depth, x_ir, p
