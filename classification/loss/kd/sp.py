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

        loss = F.mse_loss(norm_G_s, norm_G_t,reduction='none')
        # loss = F.mse_loss(norm_G_s, norm_G_t)


        return loss


def PCA_svd(X, k, center=True):
    n = X.size()[0]
    ones = torch.ones(n).view([n, 1])
    h = ((1 / n) * torch.mm(ones, ones.t())) if center else torch.zeros(n * n).view([n, n])
    H = torch.eye(n) - h
    H = H.cuda()
    X_center = torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components = v[:k].t()
    components = components.float()
    # explained_variance = torch.mul(s[:k], s[:k])/(n-1)
    return components


class DAD(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(DAD, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        fm_t = fm_t.view(fm_t.size(0), -1)

        # fm_s = PCA_svd(fm_s, 512)
        # fm_t = PCA_svd(fm_t, 512)

        fm_s_factors = torch.sqrt(torch.sum(fm_s * fm_s, 1))
        fm_s_trans = fm_s.t()
        fm_s_trans_factors = torch.sqrt(torch.sum(fm_s_trans * fm_s_trans, 0))
        # print(fm_s.shape,fm_s_factors.shape,fm_s_trans_factors.shape)
        fm_s_normal_factors = torch.mm(fm_s_factors.unsqueeze(1), fm_s_trans_factors.unsqueeze(0))
        G_s = torch.mm(fm_s, fm_s.t())
        G_s =(G_s / fm_s_normal_factors)

        fm_t_factors = torch.sqrt(torch.sum(fm_t * fm_t, 1))
        fm_t_trans = fm_t.t()
        fm_t_trans_factors = torch.sqrt(torch.sum(fm_t_trans * fm_t_trans, 0))
        fm_t_normal_factors = torch.mm(fm_t_factors.unsqueeze(1), fm_t_trans_factors.unsqueeze(0))
        G_t = torch.mm(fm_t, fm_t.t())
        G_t = (G_t / fm_t_normal_factors)

        loss = F.mse_loss(G_s, G_t)

        return loss


class DAD_MA(nn.Module):
    '''
    Similarity-Preserving Knowledge Distillation
    https://arxiv.org/pdf/1907.09682.pdf
    '''

    def __init__(self):
        super(DAD_MA, self).__init__()

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        fm_s_factors = torch.sqrt(torch.sum(fm_s * fm_s, 1))
        fm_s_factors = fm_s_factors.unsqueeze(1)
        # print(11111111111111111111111111111111111111111111)
        #
        # print(fm_s.shape, fm_s_factors.shape)

        # fm_s = fm_s / fm_s_factors
        G_s = np.zeros((fm_s.size(0), fm_s.size(0)))
        for i in range(fm_s.size(0)):
            for j in range(fm_s.size(0)):
                c = fm_s[i, :].unsqueeze(1)
                d = fm_s[j, :].unsqueeze(1)
                G_s[i, j] = F.mse_loss(c, d)

        fm_t = fm_t.view(fm_t.size(0), -1)
        fm_t_factors = torch.sqrt(torch.sum(fm_t * fm_t, 1))
        fm_t_factors = fm_t_factors.unsqueeze(1)
        # fm_t = fm_t / fm_t_factors
        G_t = np.zeros((fm_t.size(0), fm_t.size(0)))
        for i in range(fm_t.size(0)):
            for j in range(fm_t.size(0)):
                c = fm_t[i, :].unsqueeze(1)
                d = fm_t[j, :].unsqueeze(1)
                G_s[i, j] = F.mse_loss(c, d)

        G_s = torch.from_numpy(G_s).float()
        G_t = torch.from_numpy(G_t).float()
        loss = F.mse_loss(G_s, G_t)

        return loss
