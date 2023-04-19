# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch.nn as nn

from src.models.model_utils import SqueezeAndExcitation
import torch


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_depth = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, depth):
        rgb = self.se_rgb(rgb)
        depth = self.se_depth(depth)
        out = rgb + depth
        return out


class Cross_Translation(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(Cross_Translation, self).__init__()

        # self.se_m1 = SqueezeAndExcitation(channels_in,
        #                                   activation=activation)
        #
        # self.se_m2 = SqueezeAndExcitation(channels_in,
        #                                   activation=activation)

        self.transfer_m1 = nn.Sequential(nn.Conv2d(channels_in, channels_in // 2, kernel_size=1, stride=1),
                                         activation)

        self.transfer_m2 = nn.Sequential(nn.Conv2d(channels_in, channels_in // 2, kernel_size=1, stride=1),
                                         activation)

        self.transfer_m1_m2 = nn.Sequential(nn.Conv2d(channels_in, channels_in, kernel_size=1, stride=1),
                                            activation)

    def forward(self, m1_feature, m2_feature):

        # m1_feature = self.se_m1(m1_feature)
        # m2_feature = self.se_m2(m2_feature)

        m1_feature_m1 = self.transfer_m1(m1_feature)
        m2_feature_m2 = self.transfer_m2(m2_feature)

        m2_feature_m1 = self.transfer_m1(m2_feature)
        m1_feature_m2 = self.transfer_m2(m1_feature)

        joint_1 = torch.cat(((m1_feature_m1 + m2_feature_m1) / 2, (m2_feature_m2 + m1_feature_m2) / 2), dim=1)
        joint_2 = torch.cat((m1_feature_m1, m1_feature_m2), dim=1)
        joint_3 = torch.cat((m2_feature_m1, m2_feature_m2), dim=1)

        m1_m2_feature = self.transfer_m1_m2(joint_1)

        return m1_m2_feature, joint_2, joint_3
