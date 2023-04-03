from math import sqrt
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    '''
    通道注意力模块
    '''

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    '''
    空间注意力模块
    '''

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Flatten(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.flatten(x, self.dim)
        return x


import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, args, input_dims):
        super(Discriminator, self).__init__()
        self.args = args
        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            # nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        self.normal_pooling = nn.AdaptiveAvgPool2d((4, 4))
        self.pooling_2x2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pooling_1x1 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x_normal = self.normal_pooling(x)
        x_2x2 = self.pooling_2x2(x_normal)
        x_1x1 = self.pooling_1x1(x_normal)

        x_normal_flatten = torch.flatten(x_normal, start_dim=2, end_dim=3)  # B X C X feature_num

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=3)

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=3)

        x_feature = torch.cat((x_normal_flatten, x_2x2_flatten, x_1x1_flatten), dim=2)
        # print(x_feature.shape)

        # normal
        # x_feature_norm = torch.sqrt(torch.sum(x_feature ** 2, dim=1, keepdim=True))
        # x_feature = x_feature / (x_feature_norm + 1e-6)
        # x_feature[x_feature != x_feature] = 0

        return x_feature


class SPP3D(nn.Module):
    def __init__(self):
        super(SPP3D, self).__init__()
        self.normal_pooling = nn.AdaptiveAvgPool3d((2, 4, 4))
        self.pooling_2x2 = nn.AdaptiveAvgPool3d((1, 2, 2))
        self.pooling_1x1 = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x_normal = self.normal_pooling(x)
        x_2x2 = self.pooling_2x2(x_normal)
        x_1x1 = self.pooling_1x1(x_normal)

        x_normal_flatten = torch.flatten(x_normal, start_dim=2, end_dim=4)  # B X C X feature_num

        x_2x2_flatten = torch.flatten(x_2x2, start_dim=2, end_dim=4)

        x_1x1_flatten = torch.flatten(x_1x1, start_dim=2, end_dim=4)

        x_feature = torch.cat((x_normal_flatten, x_2x2_flatten, x_1x1_flatten), dim=2)

        # normal
        # x_feature_norm = torch.sqrt(torch.sum(x_feature ** 2, dim=1, keepdim=True))
        # x_feature = x_feature / (x_feature_norm + 1e-6)
        # x_feature[x_feature != x_feature] = 0

        return x_feature


class SelfAttention(nn.Module):
    dim_in: int
    dim_k: int
    dim_v: int

    def __init__(self, dim_in, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, q, x):
        # x: batch, n, dim_in


        q = self.linear_q(q)  # batch, n, dim_k
        k = self.linear_k(q)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        k = torch.transpose(k, 1, 2)

        dist = torch.bmm(q, k)  # batch, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        att = torch.bmm(dist, v)
        return att
