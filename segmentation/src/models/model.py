# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.resnet import ResNet18, ResNet34, ResNet50
from src.models.rgb_depth_fusion import SqueezeAndExciteFusionAdd, Cross_Translation
from src.models.context_modules import get_context_module
from src.models.resnet import BasicBlock, NonBottleneck1D
from src.models.model_utils import ConvBNAct, Swish, Hswish

import numpy as np


def modality_drop(x_rgb, x_depth, p):
    modality_combination = [[1, 0], [0, 1], [1, 1]]
    index_list = [x for x in range(3)]

    if p == [0, 0]:
        p = []

        # for i in range(x_rgb.shape[0]):
        #     index = random.randint(0, 6)
        #     p.append(modality_combination[index])
        #     if 'model_arch_index' in args.writer_dicts.keys():
        #         args.writer_dicts['model_arch_index'].write(str(index) + " ")
        prob = np.array((1 / 3, 1 / 3, 1 / 3))
        for i in range(x_rgb.shape[0]):
            index = np.random.choice(index_list, size=1, replace=True, p=prob)[0]
            p.append(modality_combination[index])
            # if 'model_arch_index' in args.writer_dicts.keys():
            #     args.writer_dicts['model_arch_index'].write(str(index) + " ")

        # if [0, 1] not in p:
        #     p[0] = [0, 1]
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
        p = np.array(p).reshape(x_rgb.shape[0], 2)
        p = torch.from_numpy(p)
        p = torch.unsqueeze(p, 2)
        p = torch.unsqueeze(p, 3)
        p = torch.unsqueeze(p, 4)

    p = p.float().cuda()

    x_rgb = x_rgb * p[:, 0]
    x_depth = x_depth * p[:, 1]

    return x_rgb, x_depth, p


class ESANet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear', args=None):

        super(ESANet, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_depth. Got {}'.format(encoder_rgb))

        self.channels_decoder_in = self.encoder_rgb.down_32_channels_out

        if fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_32_channels_out,
                activation=self.activation)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

        self.p = args.p
        self.modality_combination = [[True, False], [False, True], [True, True]]

    def forward(self, rgb, depth):

        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        x_rgb, x_depth, p = modality_drop(x_rgb=rgb, x_depth=depth, p=self.p)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer0(x_rgb, x_depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        x_rgb = rgb * p[:, 0]
        x_depth = depth * p[:, 1]

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer1(x_rgb, x_depth)
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        x_rgb = rgb * p[:, 0]
        x_depth = depth * p[:, 1]
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer2(x_rgb, x_depth)
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        x_rgb = rgb * p[:, 0]
        x_depth = depth * p[:, 1]
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer3(x_rgb, x_depth)
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        x_rgb = rgb * p[:, 0]
        x_depth = depth * p[:, 1]
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer4(x_rgb, x_depth)

        out = self.context_module(fuse)

        return self.decoder(enc_outs=[out, skip3, skip2, skip1])


class ESANet_KD_Auxi(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear', args=None):

        super(ESANet_KD_Auxi, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_depth. Got {}'.format(encoder_rgb))

        self.channels_decoder_in = self.encoder_rgb.down_32_channels_out

        if fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_32_channels_out,
                activation=self.activation)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

        self.p = args.p
        self.modality_combination = [[True, False], [False, True], [True, True]]

        if args.auxi:
            self.auxi_decoder = Decoder(
                channels_in=channels_after_context_module,
                channels_decoder=channels_decoder,
                activation=self.activation,
                nr_decoder_blocks=nr_decoder_blocks,
                encoder_decoder_fusion=encoder_decoder_fusion,
                upsampling_mode=upsampling,
                num_classes=num_classes
            )
        self.auxi = args.auxi

    def forward(self, rgb, depth):

        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        x_rgb, x_depth, p = modality_drop(x_rgb=rgb, x_depth=depth, p=self.p)
        p = p.float()

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer0(x_rgb, x_depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        x_rgb = rgb * p[:, 0]
        x_depth = depth * p[:, 1]

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer1(x_rgb, x_depth)
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        x_rgb = rgb * p[:, 0]
        x_depth = depth * p[:, 1]
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer2(x_rgb, x_depth)
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        x_rgb = rgb * p[:, 0]
        x_depth = depth * p[:, 1]
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer3(x_rgb, x_depth)
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        x_rgb = rgb * p[:, 0]
        x_depth = depth * p[:, 1]
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = x_rgb + x_depth
        else:
            fuse = self.se_layer4(x_rgb, x_depth)

        out = self.context_module(fuse)

        fuse_cache = (out, skip3, skip2, skip1)

        out_decoder = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        if self.auxi and (torch.sum((1 - p[:, 0]) * p[:, 1]) != 0):
            auxi_out_decoder = self.auxi_decoder(enc_outs=[out, skip3, skip2, skip1])
        else:
            auxi_out_decoder = out_decoder

        if self.training:
            if self.auxi:
                return out_decoder, fuse_cache, auxi_out_decoder, p
            else:
                return out_decoder, fuse_cache
        else:
            return out_decoder


class ESANet_DAD(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):

        super(ESANet_DAD, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_depth. Got {}'.format(encoder_rgb))

        self.channels_decoder_in = self.encoder_rgb.down_32_channels_out

        if fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_32_channels_out,
                activation=self.activation)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

    def forward(self, rgb, depth):
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer0(rgb, depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer1(rgb, depth)
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer2(rgb, depth)
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer3(rgb, depth)
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        else:
            fuse = self.se_layer4(rgb, depth)

        out = self.context_module(fuse)
        out, out_down_8_feature, out_down_16_feature, out_down_32_feature = self.decoder(
            enc_outs=[out, skip3, skip2, skip1])

        return out, out_down_8_feature, out_down_16_feature, out_down_32_feature


class ESANet_OS(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 num_classes=37,
                 encoder_rgb='resnet18',
                 encoder_depth='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=None,  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=None,  # default: [1, 1, 1]
                 fuse_depth_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):

        super(ESANet_OS, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_depth_in_rgb_encoder = fuse_depth_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_depth == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'resnet18':
            self.encoder_rgb = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet34':
            self.encoder_rgb = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation)
        elif encoder_rgb == 'resnet50':
            self.encoder_rgb = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # depth encoder
        if encoder_depth == 'resnet18':
            self.encoder_depth = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                # input_channels=1
            )
        elif encoder_depth == 'resnet34':
            self.encoder_depth = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                # input_channels=1
            )
        elif encoder_depth == 'resnet50':
            self.encoder_depth = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                # input_channels=1
            )
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        self.channels_decoder_in = self.encoder_rgb.down_32_channels_out

        if fuse_depth_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                self.encoder_rgb.down_32_channels_out,
                activation=self.activation)

        if fuse_depth_in_rgb_encoder == 'cross_tranlation':
            self.cross_layer0 = Cross_Translation(
                64, activation=self.activation)
            self.cross_layer1 = Cross_Translation(
                self.encoder_rgb.down_4_channels_out,
                activation=self.activation)
            self.cross_layer2 = Cross_Translation(
                self.encoder_rgb.down_8_channels_out,
                activation=self.activation)
            self.cross_layer3 = Cross_Translation(
                self.encoder_rgb.down_16_channels_out,
                activation=self.activation)
            self.cross_layer4 = Cross_Translation(
                self.encoder_rgb.down_32_channels_out,
                activation=self.activation)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

            layers_skip1_rgb_hall = list()
            if self.encoder_rgb.down_4_channels_out != channels_decoder[2]:
                layers_skip1_rgb_hall.append(ConvBNAct(
                    self.encoder_rgb.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1_rgb_hall = nn.Sequential(*layers_skip1_rgb_hall)

            layers_skip2_rgb_hall = list()
            if self.encoder_rgb.down_8_channels_out != channels_decoder[1]:
                layers_skip2_rgb_hall.append(ConvBNAct(
                    self.encoder_rgb.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2_rgb_hall = nn.Sequential(*layers_skip2_rgb_hall)

            layers_skip3_rgb_hall = list()
            if self.encoder_rgb.down_16_channels_out != channels_decoder[0]:
                layers_skip3_rgb_hall.append(ConvBNAct(
                    self.encoder_rgb.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3_rgb_hall = nn.Sequential(*layers_skip3_rgb_hall)


        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        self.context_module_rgb_hall, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

        self.decoder_rgb_hall = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=num_classes
        )

    def forward(self, rgb, depth):
        rgb = self.encoder_rgb.forward_first_conv(rgb)
        depth = self.encoder_depth.forward_first_conv(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        elif self.fuse_depth_in_rgb_encoder == 'SE-add':
            fuse = self.se_layer0(rgb, depth)
        else:
            fuse, layer0_joint2, layer0_joint3 = self.cross_layer0(rgb, depth)

        rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        depth = F.max_pool2d(depth, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb)
        depth = self.encoder_depth.forward_layer1(depth)
        skip_depth_1 = self.skip_layer1(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        elif self.fuse_depth_in_rgb_encoder == 'SE-add':
            fuse = self.se_layer1(rgb, depth)
        else:
            fuse, layer1_joint2, layer1_joint3 = self.cross_layer1(rgb, depth)

        skip_rgb_hall_1 = self.skip_layer1_rgb_hall(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse)
        depth = self.encoder_depth.forward_layer2(depth)
        depth_2 = depth
        skip_depth_2 = self.skip_layer2(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        elif self.fuse_depth_in_rgb_encoder == 'SE-add':
            fuse = self.se_layer2(rgb, depth)
        else:
            fuse, layer2_joint2, layer2_joint3 = self.cross_layer2(rgb, depth)
        skip_rgb_hall_2 = self.skip_layer2_rgb_hall(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse)
        depth = self.encoder_depth.forward_layer3(depth)
        depth_3 = depth
        skip_depth_3 = self.skip_layer3(depth)
        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        elif self.fuse_depth_in_rgb_encoder == 'SE-add':
            fuse = self.se_layer3(rgb, depth)
        else:
            fuse, layer3_joint2, layer3_joint3 = self.cross_layer3(rgb, depth)
        skip_rgb_hall_3 = self.skip_layer3_rgb_hall(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse)
        depth = self.encoder_depth.forward_layer4(depth)

        if self.fuse_depth_in_rgb_encoder == 'add':
            fuse = rgb + depth
        elif self.fuse_depth_in_rgb_encoder == 'SE-add':
            fuse = self.se_layer4(rgb, depth)
        else:
            fuse, layer4_joint2, layer4_joint3 = self.cross_layer4(rgb, depth)

        out_depth = self.context_module(depth)
        depth_4 = out_depth

        out_rgb_hall = self.context_module_rgb_hall(fuse)

        if self.training:
            out_depth, out_down_8_depth, out_down_16_depth, out_down_32_depth, out_down_8_feature_depth, out_down_16_feature_depth, out_down_32_feature_depth = self.decoder(
                enc_outs=[out_depth, skip_depth_3, skip_depth_2, skip_depth_1])
            out, out_down_8, out_down_16, out_down_32, out_down_8_feature, out_down_16_feature, out_down_32_feature = self.decoder_rgb_hall(
                enc_outs=[out_rgb_hall, skip_rgb_hall_3, skip_rgb_hall_2, skip_rgb_hall_1])

            # 4 ce/ 4 logits loss /3 mmd loss/ 8 rec loss
            return out, out_down_8, out_down_16, out_down_32, out_depth, out_down_8_depth, out_down_16_depth, out_down_32_depth, depth_4, depth_3, depth_2, layer0_joint2, layer0_joint3, layer1_joint2, layer1_joint3, layer2_joint2, layer2_joint3, layer3_joint2, layer3_joint3, layer4_joint2, layer4_joint3
        else:
            out, out_down_8_feature, out_down_16_feature, out_down_32_feature = self.decoder_rgb_hall(
                enc_outs=[out_rgb_hall, skip_rgb_hall_3, skip_rgb_hall_2, skip_rgb_hall_1])
            return out, out_down_8_feature, out_down_16_feature, out_down_32_feature,


class Decoder(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        self.decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )
        out_channels = channels_decoder[2]

        self.conv_out = nn.Conv2d(out_channels,
                                  num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)

    def forward(self, enc_outs):
        # out_down_feature: fused feature
        # out_down side out
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        out, out_down_32 = self.decoder_module_1(enc_out, enc_skip_down_16)
        out_down_32_feature = out
        out, out_down_16 = self.decoder_module_2(out, enc_skip_down_8)
        out_down_16_feature = out
        out, out_down_8 = self.decoder_module_3(out, enc_skip_down_4)
        out_down_8_feature = out
        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        if self.training:
            return out, out_down_8, out_down_16, out_down_32, out_down_8_feature, out_down_16_feature, out_down_32_feature
        return out, out_down_8_feature, out_down_16_feature, out_down_32_feature


class DecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)

        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec,
                                     num_classes,
                                     kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        if self.encoder_decoder_fusion == 'add':
            out += encoder_features

        return out, out_side


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2] * 2), int(x.shape[3] * 2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


def main():
    height = 480
    width = 640

    model = ESANet(
        height=height,
        width=width)

    print(model)

    model.eval()
    rgb_image = torch.randn(1, 3, height, width)
    depth_image = torch.randn(1, 1, height, width)

    with torch.no_grad():
        output = model(rgb_image, depth_image)
    print(output.shape)


if __name__ == '__main__':
    main()
