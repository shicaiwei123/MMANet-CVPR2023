from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist


class SP(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(SP, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss


class MAD(nn.Module):
    def __init__(self):
        super(MAD, self).__init__()

    def forward(self, fm_s, fm_t, logit_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        fm_t = fm_t.view(fm_t.size(0), -1)

        fm_s_trans = fm_s.t()
        fm_s_trans_list = []
        fm_s_trans_list.append(fm_s_trans[:, 0:(fm_s_trans.shape[1] // 2)])
        fm_s_trans_list.append(fm_s_trans[:, (fm_s_trans.shape[1] // 2):fm_s_trans.shape[1]])

        fm_t_trans = fm_t.t()
        fm_t_trans_list = []
        fm_t_trans_list.append(fm_t_trans[:, 0:(fm_t_trans.shape[1] // 2)])
        fm_t_trans_list.append(fm_t_trans[:, (fm_t_trans.shape[1] // 2):fm_t_trans.shape[1]])

        loss_sum = torch.zeros(fm_t_trans.shape[1]).cuda()

        for index in range(len(fm_t_trans_list)):
            G_s = torch.mm(fm_s, fm_s_trans_list[index])
            norm_G_s = F.normalize(G_s, p=2, dim=1)

            G_t = torch.mm(fm_t, fm_t_trans_list[index])
            norm_G_t = F.normalize(G_t, p=2, dim=1)

            loss = F.mse_loss(norm_G_s, norm_G_t, reduction='none')
            loss = torch.sum(loss, dim=1)
            # print(loss.shape)
            loss_sum += loss


        logit_t_prob = F.softmax(logit_t, dim=1)
        H_teacher = torch.sum(-logit_t_prob * torch.log(logit_t_prob), dim=1)
        H_teacher_prob = H_teacher / torch.sum(H_teacher)
        loss = torch.sum(loss_sum * H_teacher_prob)

        return loss
