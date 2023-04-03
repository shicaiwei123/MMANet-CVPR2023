import sys

sys.path.append('..')
from models.surf_baseline import SURF_Baseline, SURF_Multi, SURF_MV, SURF_MMANet
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from lib.model_develop import calc_accuracy_kd_patch_feature
from datasets.surf_txt import SURF, SURF_generate
from configuration.config_feature_kd import args
import torch
import torch.nn as nn
import os
import numpy as np


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "../data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'
    surf_dataset = SURF(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_multi_transforms_test, miss_modal=args.miss_modal, times=1)
    #
    # surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
    #                              transform=surf_multi_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4)

    result, _ = calc_accuracy_kd_patch_feature(model=model, loader=test_loader, args=args, verbose=True, hter=True)
    print(result)
    return result


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]

    result_model = []
    print(1)

    for i in range(3):
        i = i + 6
        result_list = []

        pretrain_dir1 = "../output/models/surf_mad_auxi_weak_version_" + str(i) + "_mad_30.0_mar_0.5.pth"
        args.gpu = 0
        args.modal = 'multi'
        args.miss_modal = 0
        args.backbone = "resnet18_se"
        args.inplace_new = 384
        args.transformer = 0
        args.buffer = 1

        for j in range(len(modality_combination)):
            args.p = modality_combination[j]
            print(args.p)

            model = SURF_MMANet(args)
            try:
                model.load_state_dict(torch.load(pretrain_dir1))
                print(pretrain_dir1)
            except Exception as e:
                # model.load_state_dict(torch.load(pretrain_dir2))
                # print(pretrain_dir2)
                sys.exit(1)

            result = batch_test(model=model, args=args)
            result_list.append(result)

        result_arr = np.array((result_list))
        print(np.mean(result_arr, axis=0))
        result_model.append(np.mean(result_arr, axis=0))
    result_model = np.array((result_model))
    print(np.mean(result_model, axis=0))
