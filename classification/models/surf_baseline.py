import torch.nn as nn
import torchvision.models as tm
import torch

from models.resnet18_se import resnet18_se
from lib.model_arch_utils import Flatten
import numpy as np
import random
from lib.model_arch import modality_drop, unbalance_modality_drop


class SURF_Multi(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)
        x_ir = self.special_bone_ir(img_ir)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4


class SURF_Baseline(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_depth = self.special_bone_depth(img_depth)
        x_ir = self.special_bone_ir(img_ir)


        # print(self.drop_mode)

        if self.drop_mode == 'average':
            # print(1)
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            # print(2)
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)

        # print(p)
        #
        # if self.drop_mode == 'average':
        #     img_rgb, img_ir, img_depth, p = modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        # else:
        #     img_rgb, img_ir, img_depth, p = unbalance_modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        #
        # print(torch.sum(img_rgb), torch.sum(img_ir), torch.sum(img_depth))
        #
        # if torch.sum(img_rgb) == 0:
        #     x_rgb = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_rgb = self.special_bone_rgb(img_rgb)
        #
        # if torch.sum(img_ir) == 0:
        #     x_ir = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_ir = self.special_bone_ir(img_ir)
        #
        # if torch.sum(img_depth) == 0:
        #     x_depth = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_depth = self.special_bone_depth(img_depth)
        #
        # print(p)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4


class SURF_Baseline_Auxi(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        args.inplace_new = 128
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(model_resnet18_se_4.layer3_new,
                                       model_resnet18_se_4.layer4,
                                       model_resnet18_se_4.avgpool,
                                       Flatten(1),
                                       model_resnet18_se_4.fc,
                                       )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        x_rgb_out = self.auxi_bone(x_rgb)
        x_ir_out = self.auxi_bone(x_ir)
        x_depth_out = self.auxi_bone(x_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        # print(p)
        #
        # if self.drop_mode == 'average':
        #     img_rgb, img_ir, img_depth, p = modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        # else:
        #     img_rgb, img_ir, img_depth, p = unbalance_modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        #
        # print(torch.sum(img_rgb), torch.sum(img_ir), torch.sum(img_depth))
        #
        # if torch.sum(img_rgb) == 0:
        #     x_rgb = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_rgb = self.special_bone_rgb(img_rgb)
        #
        # if torch.sum(img_ir) == 0:
        #     x_ir = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_ir = self.special_bone_ir(img_ir)
        #
        # if torch.sum(img_depth) == 0:
        #     x_depth = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_depth = self.special_bone_depth(img_depth)
        #
        # print(p)

        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_ir_out, x_depth_out, p


class SURF_Baseline_Auxi_Weak(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        args.inplace_new = 128
        self.transformer = nn.Conv2d(128, 128, 1, 1)
        self.transformer_rgb = nn.Conv2d(128, 128, 1, 1)
        self.transformer_depth = nn.Conv2d(128, 128, 1, 1)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(model_resnet18_se_4.layer3_new,
                                       model_resnet18_se_4.layer4,
                                       model_resnet18_se_4.avgpool,
                                       Flatten(1),
                                       model_resnet18_se_4.fc,
                                       )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        x_rgb_out = self.auxi_bone(x_rgb)
        x_depth_out = self.auxi_bone(x_depth)

        x_rgb_trans = self.transformer(x_rgb)
        x_depth_trans = self.transformer(x_depth)

        x_rgb_depth = (x_rgb_trans + x_depth_trans) / 2
        x_rgb_depth = self.auxi_bone(x_rgb_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        # print(p)
        #
        # if self.drop_mode == 'average':
        #     img_rgb, img_ir, img_depth, p = modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        # else:
        #     img_rgb, img_ir, img_depth, p = unbalance_modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        #
        # print(torch.sum(img_rgb), torch.sum(img_ir), torch.sum(img_depth))
        #
        # if torch.sum(img_rgb) == 0:
        #     x_rgb = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_rgb = self.special_bone_rgb(img_rgb)
        #
        # if torch.sum(img_ir) == 0:
        #     x_ir = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_ir = self.special_bone_ir(img_ir)
        #
        # if torch.sum(img_depth) == 0:
        #     x_depth = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_depth = self.special_bone_depth(img_depth)
        #
        # print(p)

        x = torch.cat((x_rgb, x_ir, x_depth), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p


class SURF_Baseline_Auxi_Weak_Layer4(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(
            model_resnet18_se_4.layer3_new,
            model_resnet18_se_4.layer4,
            model_resnet18_se_4.avgpool,
            Flatten(1),
            model_resnet18_se_4.fc,
        )

        # if args.buffer:
        #     self.auxi_bone = nn.Sequential(
        #         nn.Conv2d(args.inplace_new,args.inplace_new,1,1),
        #         model_resnet18_se_4.layer3_new,
        #         model_resnet18_se_4.layer4,
        #         model_resnet18_se_4.avgpool,
        #         Flatten(1),
        #         model_resnet18_se_4.fc,
        #     )
        # else:
        #     self.auxi_bone = nn.Sequential(
        #         model_resnet18_se_4.layer3_new,
        #         model_resnet18_se_4.layer4,
        #         model_resnet18_se_4.avgpool,
        #         Flatten(1),
        #         model_resnet18_se_4.fc,
        #     )

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)

        # print(p)
        #
        # if self.drop_mode == 'average':
        #     img_rgb, img_ir, img_depth, p = modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        # else:
        #     img_rgb, img_ir, img_depth, p = unbalance_modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        #
        # print(torch.sum(img_rgb), torch.sum(img_ir), torch.sum(img_depth))
        #
        # if torch.sum(img_rgb) == 0:
        #     x_rgb = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_rgb = self.special_bone_rgb(img_rgb)
        #
        # if torch.sum(img_ir) == 0:
        #     x_ir = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_ir = self.special_bone_ir(img_ir)
        #
        # if torch.sum(img_depth) == 0:
        #     x_depth = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_depth = self.special_bone_depth(img_depth)
        #
        # print(p)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)

        x_rgb_out = self.auxi_bone(x)
        x_rgb_depth = self.auxi_bone(x)
        x_depth_out = self.auxi_bone(x)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p

class SURF_MMANet(nn.Module):
    def __init__(self, args):
        super().__init__()

        args.inplace_new = 384
        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        self.auxi_bone = nn.Sequential(
            model_resnet18_se_4.layer3_new,
            model_resnet18_se_4.layer4,
            model_resnet18_se_4.avgpool,
            Flatten(1),
            model_resnet18_se_4.fc,
        )

        # if args.buffer:
        #     self.auxi_bone = nn.Sequential(
        #         nn.Conv2d(args.inplace_new,args.inplace_new,1,1),
        #         model_resnet18_se_4.layer3_new,
        #         model_resnet18_se_4.layer4,
        #         model_resnet18_se_4.avgpool,
        #         Flatten(1),
        #         model_resnet18_se_4.fc,
        #     )
        # else:
        #     self.auxi_bone = nn.Sequential(
        #         model_resnet18_se_4.layer3_new,
        #         model_resnet18_se_4.layer4,
        #         model_resnet18_se_4.avgpool,
        #         Flatten(1),
        #         model_resnet18_se_4.fc,
        #     )

    def forward(self, img_rgb, img_depth, img_ir):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_depth, x_ir, p = modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)
        else:
            x_rgb, x_depth, x_ir, p = unbalance_modality_drop(x_rgb, x_depth, x_ir, self.p, self.args)

        # print(p)
        #
        # if self.drop_mode == 'average':
        #     img_rgb, img_ir, img_depth, p = modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        # else:
        #     img_rgb, img_ir, img_depth, p = unbalance_modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        #
        # print(torch.sum(img_rgb), torch.sum(img_ir), torch.sum(img_depth))
        #
        # if torch.sum(img_rgb) == 0:
        #     x_rgb = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_rgb = self.special_bone_rgb(img_rgb)
        #
        # if torch.sum(img_ir) == 0:
        #     x_ir = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_ir = self.special_bone_ir(img_ir)
        #
        # if torch.sum(img_depth) == 0:
        #     x_depth = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_depth = self.special_bone_depth(img_depth)
        #
        # print(p)

        x = torch.cat((x_rgb, x_depth, x_ir), dim=1)
        layer3 = self.shared_bone[0](x)

        x_rgb_out = self.auxi_bone(x)
        x_rgb_depth = self.auxi_bone(x)
        x_depth_out = self.auxi_bone(x)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)

        # print(x.shape)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p


class SURF_MV(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]

        # if self.drop_mode == 'average':
        #     img_rgb, img_ir, img_depth, p = modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        # else:
        #     img_rgb, img_ir, img_depth, p = unbalance_modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        #
        # if torch.sum(img_rgb) == 0:
        #     x_rgb = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_rgb = self.special_bone_rgb(img_rgb)
        #
        # if torch.sum(img_ir) == 0:
        #     x_ir = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_ir = self.special_bone_ir(img_ir)
        #
        # if torch.sum(img_depth) == 0:
        #     x_depth = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_depth = self.special_bone_depth(img_depth)

        x_mean = (x_rgb + x_ir + x_depth) / torch.sum(p, dim=[1])

        # print(torch.sum((p)))

        x_var = torch.zeros_like(x_mean)
        if torch.sum((p)) == 1:
            x_var = torch.zeros_like(x_mean)
        else:
            for i in range(3):
                x_var += (x[i] - x_mean) ** 2
            x_var = x_var / torch.sum(p, dim=[1])
            p_sum = torch.sum(p, dim=[1, 2, 3, 4])
            # print(p_sum)
            x_var[p_sum == 1, :, :, :] = 0

        # print(torch.sum(x_mean), torch.sum(x_var))

        x_mean = x_mean.float().cuda()
        x_var = x_var.float().cuda()
        x = torch.cat((x_mean, x_var), dim=1)
        layer3 = self.shared_bone[0](x)
        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4


class SURF_MV_Auxi_Weak(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_resnet18_se_1 = resnet18_se(args, pretrained=False)
        model_resnet18_se_2 = resnet18_se(args, pretrained=False)
        model_resnet18_se_3 = resnet18_se(args, pretrained=False)

        model_resnet18_se_4 = resnet18_se(args, pretrained=False)
        self.p = args.p
        self.drop_mode = args.drop_mode
        self.args = args

        self.special_bone_rgb = nn.Sequential(model_resnet18_se_1.conv1,
                                              model_resnet18_se_1.bn1,
                                              model_resnet18_se_1.relu,
                                              model_resnet18_se_1.maxpool,
                                              model_resnet18_se_1.layer1,
                                              model_resnet18_se_1.layer2,
                                              model_resnet18_se_1.se_layer)
        self.special_bone_ir = nn.Sequential(model_resnet18_se_2.conv1,
                                             model_resnet18_se_2.bn1,
                                             model_resnet18_se_2.relu,
                                             model_resnet18_se_2.maxpool,
                                             model_resnet18_se_2.layer1,
                                             model_resnet18_se_2.layer2,
                                             model_resnet18_se_2.se_layer)
        self.special_bone_depth = nn.Sequential(model_resnet18_se_3.conv1,
                                                model_resnet18_se_3.bn1,
                                                model_resnet18_se_3.relu,
                                                model_resnet18_se_3.maxpool,
                                                model_resnet18_se_3.layer1,
                                                model_resnet18_se_3.layer2,
                                                model_resnet18_se_3.se_layer)

        self.shared_bone = nn.Sequential(model_resnet18_se_1.layer3_new,
                                         model_resnet18_se_1.layer4,
                                         model_resnet18_se_1.avgpool,
                                         Flatten(1),
                                         model_resnet18_se_1.fc,
                                         model_resnet18_se_1.dropout,
                                         )

        if args.buffer:
            self.auxi_bone = nn.Sequential(
                nn.Conv2d(args.inplace_new, args.inplace_new, 1, 1),
                model_resnet18_se_4.layer3_new,
                model_resnet18_se_4.layer4,
                model_resnet18_se_4.avgpool,
                Flatten(1),
                model_resnet18_se_4.fc,
            )
        else:
            self.auxi_bone = nn.Sequential(
                model_resnet18_se_4.layer3_new,
                model_resnet18_se_4.layer4,
                model_resnet18_se_4.avgpool,
                Flatten(1),
                model_resnet18_se_4.fc,
            )

    def forward(self, img_rgb, img_ir, img_depth):
        x_rgb = self.special_bone_rgb(img_rgb)
        x_ir = self.special_bone_ir(img_ir)
        x_depth = self.special_bone_depth(img_depth)

        if self.drop_mode == 'average':
            x_rgb, x_ir, x_depth, p = modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)
        else:
            x_rgb, x_ir, x_depth, p = unbalance_modality_drop(x_rgb, x_ir, x_depth, self.p, self.args)

        x = [x_rgb, x_ir, x_depth]

        # if self.drop_mode == 'average':
        #     img_rgb, img_ir, img_depth, p = modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        # else:
        #     img_rgb, img_ir, img_depth, p = unbalance_modality_drop(img_rgb, img_ir, img_depth, self.p, self.args)
        #
        # if torch.sum(img_rgb) == 0:
        #     x_rgb = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_rgb = self.special_bone_rgb(img_rgb)
        #
        # if torch.sum(img_ir) == 0:
        #     x_ir = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_ir = self.special_bone_ir(img_ir)
        #
        # if torch.sum(img_depth) == 0:
        #     x_depth = torch.zeros((img_rgb.shape[0], 128, 14, 14)).cuda()
        # else:
        #     x_depth = self.special_bone_depth(img_depth)

        x_mean = (x_rgb + x_ir + x_depth) / torch.sum(p, dim=[1])

        # print(torch.sum((p)))

        x_var = torch.zeros_like(x_mean)
        if torch.sum((p)) == 1:
            x_var = torch.zeros_like(x_mean)
        else:
            for i in range(3):
                x_var += (x[i] - x_mean) ** 2
            x_var = x_var / torch.sum(p, dim=[1])
            p_sum = torch.sum(p, dim=[1, 2, 3, 4])
            # print(p_sum)
            x_var[p_sum == 1, :, :, :] = 0

        # print(torch.sum(x_mean), torch.sum(x_var))

        x_mean = x_mean.float().cuda()
        x_var = x_var.float().cuda()
        x = torch.cat((x_mean, x_var), dim=1)
        layer3 = self.shared_bone[0](x)

        x_rgb_out = self.auxi_bone(x)
        x_rgb_depth = self.auxi_bone(x)
        x_depth_out = self.auxi_bone(x)

        layer4 = self.shared_bone[1](layer3)
        x = self.shared_bone[2](layer4)
        x = self.shared_bone[3](x)
        x = self.shared_bone[4](x)
        # x = self.shared_bone[5](x)
        return x, layer3, layer4, x_rgb_out, x_rgb_depth, x_depth_out, p

