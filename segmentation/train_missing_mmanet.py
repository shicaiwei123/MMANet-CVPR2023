# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from datetime import datetime
import json
import pickle
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model_kd_auxi
from src import utils
from src.prepare_data import prepare_data
from src.utils import save_ckpt_one_modality, save_ckpt_every_epoch_one_modality
from src.utils import load_ckpt_one_modality
from src.utils import print_log_original

from src.logger import CSVLogger
from src.confusion_matrix import ConfusionMatrixTensorflow

from torch.cuda.amp import autocast as autocast, GradScaler
import torch.nn.functional as F
from loss.kd.mad import SP, MAD

scaler = GradScaler()


def parse_args():
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()
    args.encoder = 'resnet50'
    args.p = [0, 0]
    args.begin_epoch = 10
    # args.last_ckpt = "/home/ssd/mmanet/cityscape/mad_mar/cityscapes-with-depth/checkpoints_04_04_2023-11_24_06-716365/ckpt_latest.pth"

    # The provided learning rate refers to the default batch size of 8.
    # When using different batch sizes we need to adjust the learning rate
    # accordingly:
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        warnings.warn(f'Adapting learning rate to {args.lr} because provided '
                      f'batch size differs from default batch size of 8.')

    return args


def train_main():
    args = parse_args()

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(args.results_dir, args.dataset,
                            f'checkpoints_{training_starttime}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'confusion_matrices'), exist_ok=True)

    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, 'argsv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    # when using multi scale supervision the label needs to be downsampled.
    label_downsampling_rates = [8, 16, 32]

    # data preparation ---------------------------------------------------------
    data_loaders = prepare_data(args, ckpt_dir)

    if args.dataset == 'nyuv2':
        args.class_num = 40
    else:
        args.class_num = 19

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

    args.p = [1, 1]
    args.auxi = False
    teacher, device = build_model_kd_auxi(args, n_classes=n_classes_without_void)

    args.p = [0, 0]
    args.auxi = True
    model, device = build_model_kd_auxi(args, n_classes=n_classes_without_void)

    if args.dataset == 'nyuv2':

        checkpoint = torch.load(
            "/home/ssd/mmanet/results/nyuv2/fullr50/nyuv2/checkpoints_09_03_2023-14_53_44-596878/ckpt_epoch_245.pth",
            map_location=lambda storage, loc: storage)
        teacher.load_state_dict(checkpoint['rgb_state_dict'])

    else:
        checkpoint = torch.load(
            "/home/icml/shicaiwei/ESANet-main/results/cityscapes-with-depth/full/cityscapes-with-depth/checkpoints_28_09_2022-09_36_35-226770/ckpt_epoch_284.pth",
            map_location=lambda storage, loc: storage)
        teacher.load_state_dict(checkpoint['rgb_state_dict'])

    for p in teacher.parameters():
        p.requires_grad = False

    print(teacher.auxi)

    # kd_loss = SP()
    kd_loss = MAD()

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
        args.weak_combintaions = torch.tensor([[0, 1]]).cuda()
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0

    print(start_epoch)

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

    csvlogger = CSVLogger(log_keys_for_csv, os.path.join(ckpt_dir, 'logs.csv'),
                          append=True)

    # one confusion matrix per camera and one for whole valid data
    confusion_matrices = dict()
    for camera in cameras:
        confusion_matrices[camera] = \
            ConfusionMatrixTensorflow(n_classes_without_void)
        confusion_matrices['all'] = \
            ConfusionMatrixTensorflow(n_classes_without_void)

    # start training -----------------------------------------------------------
    torch.backends.cudnn.enabled = False

    modality_combination = [[1, 0], [0, 1], [1, 1]]

    disttribution_distance = [0] * (len(modality_combination) - 1)
    disttribution_distance_mse = [0] * (len(modality_combination) - 1)

    weak_combintaions = []
    prediction_list = []

    for epoch in range(int(start_epoch), args.epochs):
        # unfreeze
        if args.freeze == epoch and args.finetune is None:
            print('Unfreezing')
            for param in model.parameters():
                param.requires_grad = True

        model.p = [0, 0]
        logs = train_one_epoch(
            model, teacher, kd_loss, train_loader, device, optimizer, loss_function_train, epoch,
            lr_scheduler, args.modality,
            label_downsampling_rates, args=args, debug_mode=args.debug)

        ### constrasting ranking
        ### For segmentation task, we measure  MIOU to select the weak modality for more robust results

        if epoch > 5 and epoch < args.begin_epoch:
            label_disttribution = []
            for j in range(len(modality_combination)):
                print(j)
                args.p = modality_combination[j]
                print(args.p)
                model.p = modality_combination[j]

                miou, logs, prediction = validate(
                    model, valid_loader, device, cameras,
                    confusion_matrices, args.modality, loss_function_valid, logs,
                    ckpt_dir, epoch, loss_function_valid_unweighted,
                    debug_mode=args.debug, get_prediction=True)


                label_disttribution.append([miou['all']])

            label_disttribution = torch.tensor(label_disttribution).float()
            print(label_disttribution)
            for i in range(len(label_disttribution) - 1):
                disttribution_distance_mse[i] = disttribution_distance_mse[i] + F.mse_loss(
                    label_disttribution[len(label_disttribution) - 1], label_disttribution[i])
                print("mse", F.mse_loss(label_disttribution[len(label_disttribution) - 1], label_disttribution[i]))

            print(disttribution_distance_mse)
            disttribution_distance = disttribution_distance_mse

        if epoch == args.begin_epoch:
            max_index = disttribution_distance.index(max(disttribution_distance))
            weak_combintaion = modality_combination[max_index]
            print(weak_combintaion)
            weak_combintaions.append(weak_combintaion)
            weak_combintaions = torch.tensor(weak_combintaions).cuda()
            args.weak_combintaions = weak_combintaions
            # sys.exit(1)

        model.p = [0, 0]

        # validation after every epoch -----------------------------------------
        if epoch >= 240:
            miou, logs = validate(
                model, valid_loader, device, cameras,
                confusion_matrices, args.modality, loss_function_valid, logs,
                ckpt_dir, epoch, loss_function_valid_unweighted,
                debug_mode=args.debug
            )

            if args.valid_full_res:
                miou_full_res, logs = validate(
                    model, valid_loader_full_res, device, cameras,
                    confusion_matrices, args.modality, loss_function_valid, logs,
                    ckpt_dir,
                    epoch, loss_function_valid_unweighted,
                    add_log_key='_full-res', debug_mode=args.debug
                )

            logs.pop('time', None)
            csvlogger.write_logs(logs)

            # save weights
            print(miou['all'])
            save_current_checkpoint = False
            if miou['all'] > best_miou:
                best_miou = miou['all']
                best_miou_epoch = epoch
                save_current_checkpoint = True

            if epoch > 290:
                save_current_checkpoint = True

            if args.valid_full_res and miou_full_res['all'] > best_miou_full_res:
                best_miou_full_res = miou_full_res['all']
                best_miou_full_res_epoch = epoch
                save_current_checkpoint = True

            # don't save weights for the first 10 epochs as mIoU is likely getting
            # better anyway
            if epoch >= 10 and save_current_checkpoint is True:
                save_ckpt_one_modality(ckpt_dir, model, optimizer, epoch)

            # save / overwrite latest weights (useful for resuming training)
        save_ckpt_every_epoch_one_modality(ckpt_dir, model, optimizer, epoch, best_miou,
                                           best_miou_epoch)

    # write a finish file with best miou values in order overview
    # training result quickly
    with open(os.path.join(ckpt_dir, 'finished.txt'), 'w') as f:
        f.write('best miou: {}\n'.format(best_miou))
        f.write('best miou epoch: {}\n'.format(best_miou_epoch))
        if args.valid_full_res:
            f.write(f'best miou full res: {best_miou_full_res}\n')
            f.write(f'best miou full res epoch: {best_miou_full_res_epoch}\n')

    print(weak_combintaions)
    print("Training completed ")


def train_one_epoch(model, teacher, kd_loss, train_loader, device, optimizer, loss_function_train,
                    epoch, lr_scheduler, modality,
                    label_downsampling_rates, args, debug_mode=False):
    training_start_time = time.time()
    lr_scheduler.step(epoch)
    samples_of_epoch = 0

    # set model to train mode
    model.train()

    for param in teacher.parameters():
        param.grad = None

    # loss for every resolution
    losses_list = []

    # summed loss of all resolutions
    total_loss_list = []

    for i, sample in enumerate(train_loader):
        start_time_for_one_step = time.time()

        # load the data and send them to gpu
        if modality in ['rgbd', 'rgb']:
            image = sample['image'].to(device)
            batch_size = image.data.shape[0]
        if modality in ['rgbd', 'depth']:
            depth = sample['depth'].to(device)
            batch_size = depth.data.shape[0]
        target_scales = [sample['label'].to(device)]
        if len(label_downsampling_rates) > 0:
            for rate in sample['label_down']:
                target_scales.append(sample['label_down'][rate].to(device))

        for param in model.parameters():
            param.grad = None

        with torch.no_grad():
            # print(teacher.auxi)
            pred_scales_t, fuse_cache_t = teacher(image, depth)

        with autocast():

            # forward pass
            if modality == 'rgbd':
                pred_scales, fuse_cache, auxi_out, p = model(image, depth)
            elif modality == 'rgb':
                pred_scales, fuse_cache, auxi_out, p = model(image)
            else:
                pred_scales, fuse_cache, auxi_out, p = model(depth)

            # loss computation
            # print(len(pred_scales), len(target_scales))
            losses = loss_function_train(pred_scales, target_scales)
            loss_segmentation = sum(losses)

            if epoch > args.begin_epoch:

                weak_index = p == args.weak_combintaions[0].unsqueeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).float()
                weak_index = weak_index.all(dim=1)

                index = [bool(ii) for ii in weak_index]
                # print(index)
                auxi_out_weak = [auxi_out[ii][index] for ii in range(len(target_scales))]
                target_scales_weak = [target_scales[ii][index] for ii in range(len(target_scales))]

                # print(target_scales_weak[0].shape[0])
                if target_scales_weak[0].shape[0] == 0:
                    loss_segmentation_weak = torch.zeros(1).cuda().float() + 1e-16
                else:
                    losses_weak = loss_function_train(auxi_out_weak, target_scales_weak)
                    loss_segmentation_weak = sum(losses_weak)
            else:
                loss_segmentation_weak = torch.zeros(1).cuda().float() + 1e-16

            # distillation loss
            # for segmentation task, we perform distillation on multi-scale encoder feature map output
            distillation_loss = torch.zeros(1).cuda().float()
            for j in [0, 1, 2]:

                fuse_cache_sample = fuse_cache[j]  #### B C H W
                fuse_cache_t_sample = fuse_cache_t[j]

                ## if the feature map size is bigger than 30, performing pooling to reduce the memory requirements of GPU
                if fuse_cache_sample.shape[2] > 30:
                    fuse_cache_sample = F.adaptive_avg_pool2d(fuse_cache_sample, (30, 40))
                    fuse_cache_t_sample = F.adaptive_avg_pool2d(fuse_cache_t_sample, (30, 40))

                fuse_cache_view = fuse_cache_sample.permute((0, 2, 3, 1))  ####B H W C
                fuse_cache_t_view = fuse_cache_t_sample.permute((0, 2, 3, 1))

                ## For segmentation task, we take each pixel of feature map as a class and the correspondding channel feature as its features.
                shape = fuse_cache_t_view.shape
                fuse_cache_t_view = fuse_cache_t_view.contiguous().view(-1, shape[3])
                fuse_cache_view = fuse_cache_view.contiguous().view(-1, shape[3])

                pred_t_pooling = F.adaptive_avg_pool2d(pred_scales_t[6],
                                                       (fuse_cache_sample.shape[2], fuse_cache_sample.shape[3]))

                pred_t_pooling = pred_t_pooling.permute((0, 2, 3, 1))
                pred_t_pooling = pred_t_pooling.contiguous().view(-1, pred_t_pooling.shape[3])

                distillation_loss_cls = kd_loss(fuse_cache_view, fuse_cache_t_view.detach(), pred_t_pooling)
                distillation_loss += distillation_loss_cls

            if torch.isnan(distillation_loss):
                print("nan")
                distillation_loss = torch.zeros(1).cuda().float()

            if torch.isnan(distillation_loss):
                print("Loss is nan")
                print(loss_segmentation, distillation_loss)
                loss_segmentation = torch.zeros(1).float().cuda()

            if epoch > args.begin_epoch:
                total_loss = loss_segmentation + distillation_loss * args.kd_scale + loss_segmentation_weak * args.auxi_scale
            else:
                total_loss = loss_segmentation

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # append loss values to the lists. Later we can calculate the
        # mean training loss of this epoch
        losses_list.append([loss.cpu().detach().numpy() for loss in losses])
        total_loss = total_loss.cpu().detach().numpy()
        total_loss_list.append(total_loss)

        # if np.isnan(total_loss):
        #     raise ValueError('Loss is None')

        # print log
        samples_of_epoch += batch_size
        time_inter = time.time() - start_time_for_one_step

        learning_rates = lr_scheduler.get_lr()

        print_log_original(epoch, samples_of_epoch, batch_size,
                           len(train_loader.dataset), total_loss, time_inter,
                           learning_rates)

        print(loss_segmentation.detach().cpu().numpy(), distillation_loss.detach().cpu().numpy(),
              loss_segmentation_weak.detach().cpu().numpy())

        if debug_mode:
            # only one batch while debugging
            break

    # fill the logs for csv log file and web logger
    logs = dict()
    logs['time_training'] = time.time() - training_start_time
    logs['loss_train_total'] = np.mean(total_loss_list)
    losses_train = np.mean(losses_list, axis=0)
    logs['loss_train_full_size'] = losses_train[0]
    for i, rate in enumerate(label_downsampling_rates):
        logs['loss_train_down_{}'.format(rate)] = losses_train[i + 1]
    logs['epoch'] = epoch
    for i, lr in enumerate(learning_rates):
        logs['lr_{}'.format(i)] = lr
    return logs


def validate(model, valid_loader, device, cameras, confusion_matrices,
             modality, loss_function_valid, logs, ckpt_dir, epoch,
             loss_function_valid_unweighted=None, add_log_key='',
             debug_mode=False, get_prediction=False):
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

                    # prediction_save = prediction

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

                    # prediction_save = prediction

                    # ignore void pixels
                    mask = label > 0
                    label = torch.masked_select(label, mask)
                    prediction = torch.masked_select(prediction,
                                                     mask.to(device))

                    # print("xxx",prediction.shape)

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

    # save the confusion matrices of this epoch.
    # This helps if we want to compute other metrics later.
    with open(os.path.join(ckpt_dir, 'confusion_matrices',
                           f'cm_epoch_{epoch}.pickle'), 'wb') as f:
        pickle.dump({k: cm.overall_confusion_matrix
                     for k, cm in confusion_matrices.items()}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # logs for the csv logger and the web logger
    logs[f'loss_{valid_split}'] = \
        loss_function_valid.compute_whole_loss()

    if loss_function_valid_unweighted is not None:
        logs[f'loss_{valid_split}_unweighted'] = \
            loss_function_valid_unweighted.compute_whole_loss()

    logs[f'mIoU_{valid_split}'] = miou['all']
    for camera in cameras:
        logs[f'mIoU_{valid_split}_{camera}'] = miou[camera]

    logs['time_validation'] = validation_time
    logs['time_confusion_matrix'] = cm_time
    logs['time_forward'] = forward_time
    logs['time_post_processing'] = post_processing_time
    logs['time_copy_to_gpu'] = copy_to_gpu_time

    # write iou value of every class to logs
    for i, iou_value in enumerate(ious['all']):
        logs[f'IoU_{valid_split}_class_{i}'] = iou_value

    if get_prediction:
        return miou, logs, prediction
    else:
        return miou, logs


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
    train_main()
