# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import sys

sys.path.append('..')
import argparse
from datetime import datetime
import json
import pickle
import os
import sys
import time
import warnings

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR

os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model_original
from src import utils
from src.prepare_data import prepare_data

from src.utils import load_ckpt_one_modality
from src.utils import print_log_original

from src.logger import CSVLogger
from src.confusion_matrix import ConfusionMatrixTensorflow


def parse_args():
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()
    args.encoder = 'resnet50'
    args.p = [1, 0]
    # args.last_ckpt="/home/icml//shicaiwei/ESANet-main/results/nyuv2/rgbde_res34/nyuv2/checkpoints_10_04_2022-10_20_09-674318/ckpt_latest.pth"

    if args.dataset != 'nyuv2':
        args.valid_full_res = True
        args.batch_size = 2
        args.valid_batch_size=2

    # The provided learning rate refers to the default batch size of 8.
    # When using different batch sizes we need to adjust the learning rate
    # accordingly:
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        warnings.warn(f'Adapting learning rate to {args.lr} because provided '
                      f'batch size differs from default batch size of 8.')

    return args


def train_main(args):
    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(args.results_dir, args.dataset,
                            f'checkpoints_{training_starttime}')

    # when using multi scale supervision the label needs to be downsampled.
    label_downsampling_rates = [8, 16, 32]

    # data preparation ---------------------------------------------------------
    data_loaders = prepare_data(args)

    if args.valid_full_res:
        train_loader, valid_loader, valid_loader_full_res = data_loaders
    else:
        train_loader, valid_loader = data_loaders
        valid_loader_full_res = None

    cameras = train_loader.dataset.cameras
    n_classes_without_void = train_loader.dataset.n_classes_without_void
    if args.class_weighting != 'None':
        class_weighting = train_loader.dataset.compute_class_weights(
            weight_mode=args.class_weighting,
            c=args.c_for_logarithmic_weighting)
    else:
        class_weighting = np.ones(n_classes_without_void)

    # model building -----------------------------------------------------------
    model, device = build_model_original(args, n_classes=n_classes_without_void)
    if args.dataset == 'nyuv2':
        # checkpoint = torch.load(
        #     "../results/nyuv2/rgbd_r50_kd_sr_amp/nyuv2/checkpoints_15_11_2022-15_06_14-866040/ckpt_epoch_293.pth",
        #     map_location=lambda storage, loc: storage)
        checkpoint = torch.load(
            "../results/nyuv2/mad_mar/nyuv2/checkpoints_19_04_2023-09_32_25-228802/ckpt_epoch_"+str(args.i)+".pth",
            map_location=lambda storage, loc: storage)

        print("../results/nyuv2/mad_mar/nyuv2/checkpoints_19_04_2023-09_32_25-228802/ckpt_epoch_"+str(args.i)+".pth",)

    elif args.dataset == 'sunrgbd':
        checkpoint = torch.load(
            "../results/sunrgbd/rgbd_r50_missing_kd_amp/sunrgbd/checkpoints_21_09_2022-11_21_28-899506/ckpt_epoch_260.pth",
            map_location=lambda storage, loc: storage)
    else:
        # checkpoint = torch.load(
        #     "../results/missing_kd_auxi_amp/cityscapes-with-depth/checkpoints_15_10_2022-16_18_20-210412/ckpt_epoch_289.pth",
        #     map_location=lambda storage, loc: storage)
        checkpoint = torch.load(
            "/home/ssd/mmanet/cityscape/mad_mar/cityscapes-with-depth/checkpoints_05_04_2023-10_02_35-036121/ckpt_epoch_"+str(args.i)+".pth",
            map_location=lambda storage, loc: storage)

        print("/home/ssd/mmanet/cityscape/mad_mar/cityscapes-with-depth/checkpoints_05_04_2023-10_02_35-036121/ckpt_epoch_"+str(args.i)+".pth")
    model.load_state_dict(checkpoint['rgb_state_dict'], strict=False)

    if args.freeze > 0:
        print('Freeze everything but the output layer(s).')
        for name, param in model.named_parameters():
            if 'out' not in name:
                param.requires_grad = False

    # loss, optimizer, learning rate scheduler, csvlogger  ----------

    # loss functions (only loss_function_train is really needed.
    # The other loss functions are just there to compare valid loss to
    # train loss)
    loss_function_train = \
        utils.CrossEntropyLoss2d(weight=class_weighting, device=device)

    pixel_sum_valid_data = valid_loader.dataset.compute_class_weights(
        weight_mode='linear'
    )
    pixel_sum_valid_data_weighted = \
        np.sum(pixel_sum_valid_data * class_weighting)
    loss_function_valid = utils.CrossEntropyLoss2dForValidData(
        weight=class_weighting,
        weighted_pixel_sum=pixel_sum_valid_data_weighted,
        device=device
    )
    loss_function_valid_unweighted = \
        utils.CrossEntropyLoss2dForValidDataUnweighted(device=device)

    optimizer = get_optimizer(args, model)

    # in this script lr_scheduler.step() is only called once per epoch
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=[i['lr'] for i in optimizer.param_groups],
        total_steps=args.epochs,
        div_factor=25,
        pct_start=0.1,
        anneal_strategy='cos',
        final_div_factor=1e4
    )

    # load checkpoint if parameter last_ckpt is provided
    if args.last_ckpt:
        ckpt_path = os.path.join(ckpt_dir, args.last_ckpt)
        epoch_last_ckpt, best_miou, best_miou_epoch = \
            load_ckpt_one_modality(model, optimizer, ckpt_path, device)
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0

    valid_split = valid_loader.dataset.split

    # build the log keys for the csv log file and for the web logger
    log_keys = [f'mIoU_{valid_split}']
    if args.valid_full_res:
        log_keys.append(f'mIoU_{valid_split}_full-res')
        best_miou_full_res = 0

    log_keys_for_csv = log_keys.copy()

    # mIoU for each camera
    for camera in cameras:
        log_keys_for_csv.append(f'mIoU_{valid_split}_{camera}')
        if args.valid_full_res:
            log_keys_for_csv.append(f'mIoU_{valid_split}_full-res_{camera}')

    log_keys_for_csv.append('epoch')
    for i in range(len(lr_scheduler.get_lr())):
        log_keys_for_csv.append('lr_{}'.format(i))
    log_keys_for_csv.extend(['loss_train_total', 'loss_train_full_size'])
    for rate in label_downsampling_rates:
        log_keys_for_csv.append('loss_train_down_{}'.format(rate))
    log_keys_for_csv.extend(['time_training', 'time_validation',
                             'time_confusion_matrix', 'time_forward',
                             'time_post_processing', 'time_copy_to_gpu'])

    valid_names = [valid_split]
    if args.valid_full_res:
        valid_names.append(valid_split + '_full-res')
    for valid_name in valid_names:
        # iou for every class
        for i in range(n_classes_without_void):
            log_keys_for_csv.append(f'IoU_{valid_name}_class_{i}')
        log_keys_for_csv.append(f'loss_{valid_name}')
        if loss_function_valid_unweighted is not None:
            log_keys_for_csv.append(f'loss_{valid_name}_unweighted')

    # one confusion matrix per camera and one for whole valid data
    confusion_matrices = dict()
    for camera in cameras:
        confusion_matrices[camera] = \
            ConfusionMatrixTensorflow(n_classes_without_void)
        confusion_matrices['all'] = \
            ConfusionMatrixTensorflow(n_classes_without_void)

    # start training -----------------------------------------------------------
    torch.backends.cudnn.enabled = False

    for epoch in range(int(start_epoch), args.epochs):
        # unfreeze
        if args.freeze == epoch and args.finetune is None:
            print('Unfreezing')
            for param in model.parameters():
                param.requires_grad = True

        # validation after every epoch -----------------------------------------
        if epoch >= 0:

            if not args.valid_full_res:
                miou = validate(
                    model, valid_loader, device, cameras,
                    confusion_matrices, args.modality, loss_function_valid,
                    ckpt_dir, epoch, loss_function_valid_unweighted,
                    debug_mode=args.debug
                )
            # miou_full_res = validate(
            #     model, valid_loader_full_res, device, cameras,
            #     confusion_matrices, args.modality, loss_function_valid,
            #     ckpt_dir,
            #     epoch, loss_function_valid_unweighted,
            #     add_log_key='_full-res', debug_mode=args.debug
            # )
            # print(args.valid_full_res)
            if args.valid_full_res:
                miou_full_res = validate(
                    model, valid_loader_full_res, device, cameras,
                    confusion_matrices, args.modality, loss_function_valid,
                    ckpt_dir,
                    epoch, loss_function_valid_unweighted,
                    add_log_key='_full-res', debug_mode=args.debug
                )
                print(miou_full_res['all'])

            # save weights
            # print(miou['all'])
            # save_current_checkpoint = False
            # if miou['all'] > best_miou:
            #     best_miou = miou['all']
            #     best_miou_epoch = epoch
            #
            # if args.valid_full_res and miou_full_res['all'] > best_miou_full_res:
            #     best_miou_full_res = miou_full_res['all']
            #     best_miou_full_res_epoch = epoch

            # don't save weights for the first 10 epochs as mIoU is likely getting
            # better anyway
        break

    # write a finish file with best miou values in order overview
    # training result quickly

    print("Training completed ")


def validate(model, valid_loader, device, cameras, confusion_matrices,
             modality, loss_function_valid, ckpt_dir, epoch,
             loss_function_valid_unweighted=None, add_log_key='',
             debug_mode=False):
    valid_split = valid_loader.dataset.split + add_log_key

    print(f'Validation on {valid_split}')

    # we want to track how long each part of the validation takes
    validation_start_time = time.time()
    cm_time = 0  # time for computing all confusion matrices
    forward_time = 0
    post_processing_time = 0
    copy_to_gpu_time = 0

    # set model to eval mode
    model.eval()

    # we want to store miou and ious for each camera
    miou = dict()
    ious = dict()

    # reset loss (of last validation) to zero
    loss_function_valid.reset_loss()

    if loss_function_valid_unweighted is not None:
        loss_function_valid_unweighted.reset_loss()

    # validate each camera after another as all images of one camera have
    # the same resolution and can be resized together to the ground truth
    # segmentation size.
    for camera in cameras:
        with valid_loader.dataset.filter_camera(camera):
            confusion_matrices[camera].reset_conf_matrix()
            print(f'{camera}: {len(valid_loader.dataset)} samples')

            for i, sample in enumerate(valid_loader):
                # copy the data to gpu
                copy_to_gpu_time_start = time.time()
                if modality in ['rgbd', 'rgb']:
                    image = sample['image'].to(device)
                if modality in ['rgbd', 'depth']:
                    depth = sample['depth'].to(device)
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                copy_to_gpu_time += time.time() - copy_to_gpu_time_start

                # forward pass
                with torch.no_grad():
                    forward_time_start = time.time()
                    if modality == 'rgbd':
                        prediction, _, _, _ = model(image, depth)
                    elif modality == 'rgb':
                        prediction, _, _, _ = model(image)
                    else:
                        prediction, _, _, _ = model(depth)
                    if not device.type == 'cpu':
                        torch.cuda.synchronize()
                    forward_time += time.time() - forward_time_start

                    # compute valid loss
                    post_processing_time_start = time.time()

                    loss_function_valid.add_loss_of_batch(
                        prediction,
                        sample['label'].to(device)
                    )

                    if loss_function_valid_unweighted is not None:
                        loss_function_valid_unweighted.add_loss_of_batch(
                            prediction, sample['label'].to(device))

                    # this label is not preprocessed and therefore still has its
                    # original size
                    label = sample['label_orig']
                    _, image_h, image_w = label.shape

                    # resize the prediction to the size of the original ground
                    # truth segmentation before computing argmax along the
                    # channel axis
                    prediction = F.interpolate(
                        prediction,
                        (image_h, image_w),
                        mode='bilinear',
                        align_corners=False)
                    prediction = torch.argmax(prediction, dim=1)

                    # ignore void pixels
                    mask = label > 0
                    label = torch.masked_select(label, mask)
                    prediction = torch.masked_select(prediction,
                                                     mask.to(device))

                    # In the label 0 is void, but in the prediction 0 is wall.
                    # In order for the label and prediction indices to match we
                    # need to subtract 1 of the label.
                    label -= 1

                    # copy the prediction to cpu as tensorflow's confusion
                    # matrix is faster on cpu
                    prediction = prediction.cpu()

                    label = label.numpy()
                    prediction = prediction.numpy()
                    post_processing_time += \
                        time.time() - post_processing_time_start

                    # finally compute the confusion matrix
                    cm_start_time = time.time()
                    # print(label,prediction)
                    confusion_matrices[camera].update_conf_matrix(label,
                                                                  prediction)
                    cm_time += time.time() - cm_start_time

                    if debug_mode:
                        # only one batch while debugging
                        break

            # After all examples of camera are passed through the model,
            # we can compute miou and ious.
            cm_start_time = time.time()
            miou[camera], ious[camera] = \
                confusion_matrices[camera].compute_miou()
            cm_time += time.time() - cm_start_time
            print(f'mIoU {valid_split} {camera}: {miou[camera]}')

    # confusion matrix for the whole split
    # (sum up the confusion matrices of all cameras)
    cm_start_time = time.time()
    confusion_matrices['all'].reset_conf_matrix()
    for camera in cameras:
        confusion_matrices['all'].overall_confusion_matrix += \
            confusion_matrices[camera].overall_confusion_matrix

    # miou and iou for all cameras
    miou['all'], ious['all'] = confusion_matrices['all'].compute_miou()
    cm_time += time.time() - cm_start_time
    print(f"mIoU {valid_split}: {miou['all']}")

    validation_time = time.time() - validation_start_time

    return miou,


def get_optimizer(args, model):
    # set different learning rates fo different parts of the model
    # when using default parameters the whole model is trained with the same
    # learning rate
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(
            'Currently only SGD and Adam as optimizers are '
            'supported. Got {}'.format(args.optimizer))

    print('Using {} as optimizer'.format(args.optimizer))
    return optimizer


if __name__ == '__main__':
    args = parse_args()

    for i in range(10):
        i=290+i

        for p in [[0,1],[1,0],[1, 1]]:
            args.p = p
            args.i=i
            try:
                train_main(args)
            except Exception as e:
                print(e)
                print(3)
