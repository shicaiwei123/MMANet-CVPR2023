from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.mmd_loss import MMD_loss
from lib.model_arch_utils import SelfAttention
import numpy as np


# nn.Linear()


class SoftTarget(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''

    def __init__(self, T):
        super(SoftTarget, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                        F.softmax(out_t / self.T, dim=1),
                        reduction='batchmean') * self.T * self.T

        return loss


class LinearWeightedAvg(nn.Module):
    def __init__(self, n_inputs):
        super(LinearWeightedAvg, self).__init__()
        self.weights = nn.ParameterList([nn.Parameter(torch.tensor(1.0) / n_inputs) for i in range(n_inputs)])

    def forward(self, input):
        res = 0
        for emb_idx, emb in enumerate(input):
            res += emb * self.weights[emb_idx]
        return res


class MultiSoftTarget(nn.Module):
    def __init__(self, T):
        super(MultiSoftTarget, self).__init__()
        self.T = T
        self.scale_selector = nn.Sequential(nn.Linear(21, 21), nn.Softmax())
        self.self_attention = SelfAttention(21, 21, 21)
        self.mmd = MMD_loss()

    def forward(self, out_s_multi, out_t_multi, weight=None):
        loss_sum = torch.tensor(0.0)

        # print(out_s_multi.shape)
        # print(out_s_multi[0, 1])
        loss_list = []
        for i in range(out_s_multi.shape[2]):
            out_s = torch.squeeze(out_s_multi[:, :, i], dim=1)
            out_t = torch.squeeze(out_t_multi[:, :, i], dim=1)

            loss = F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                            F.softmax(out_t / self.T, dim=1),
                            reduction='batchmean') * self.T * self.T
            # print(loss)
            loss_list.append(loss)
            # print(loss)
            loss_sum = loss_sum + loss

        # print(loss_list)

        if weight is not None:

            # loss_sum = torch.sum(self.scale_selector(weight) * loss_list)
            loss_list = torch.tensor(loss_list)
            loss_list=torch.unsqueeze(loss_list,0)
            loss_list = torch.unsqueeze(loss_list, 2)
            loss_list = loss_list.cuda()
            loss_sum = torch.sum(self.self_attention(weight,loss_list))

        else:
            loss_list = torch.tensor(loss_list)
            loss_list = loss_list.cuda()
            loss_sum = torch.sum(loss_list) / out_s_multi.shape[2]
        return loss_sum
