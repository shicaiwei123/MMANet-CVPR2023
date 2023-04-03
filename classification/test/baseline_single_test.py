import sys

sys.path.append('..')
import os
from models.resnet18_se import resnet18_se
from src.surf_baseline_single_dataloader import surf_single_transforms_test
from lib.model_develop_utils import calc_accuracy
from datasets.surf_single_txt import SURF_Single
from lib.model_develop import calc_accuracy_multi
from datasets.surf_txt import SURF, SURF_generate

from models.surf_baseline import SURF_Baseline
# from configuration.config_baseline_single import args
# from src.surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from src.surf_baseline_multi_dataloader import surf_multi_transforms_train, surf_multi_transforms_test
from configuration.config_baseline_multi import args

import torch
import torch.nn as nn
import torchvision.models as tm


def batch_test_single(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "../data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'
    surf_dataset = SURF_Single(txt_dir=txt_dir,
                               root_dir=root_dir,
                               transform=surf_single_transforms_test, modal=args.modal)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
    )

    result = calc_accuracy(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    root_dir = "../data/CASIA-SURF"
    txt_dir = root_dir + '/test_private_list.txt'
    surf_dataset = SURF(txt_dir=txt_dir,
                        root_dir=root_dir,
                        transform=surf_multi_transforms_test, miss_modal=args.miss_modal)
    #
    # surf_dataset = SURF_generate(rgb_dir=args.rgb_root, depth_dir=args.depth_root, ir_dir=args.ir_root,
    #                              transform=surf_multi_transforms_test)

    test_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4)

    result, _ = calc_accuracy_multi(model=model, loader=test_loader, verbose=True, hter=True)
    print(result)
    return result


def multi_single_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
    result_list = []

    moldaity_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    modality = ['rgb', 'depth', 'ir']
    modality_index = 2
    for k in range(3):
        pretrain_dir = "../output/models/multi_" + modality[modality_index] + "_" + str(k) + "__average.pth"
        print(pretrain_dir)
        args.gpu = 3
        args.modal = 'multi'
        args.miss_modal = 0
        args.p = moldaity_combination[modality_index]
        args.backbone = "resnet18_se"
        model = SURF_Baseline(args)
        test_para = torch.load(pretrain_dir)
        model.load_state_dict(torch.load(pretrain_dir))

        result = batch_test(model=model, args=args)


def single_single_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
    result_list = []

    moldaity_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    modality = ['rgb', 'depth', 'ir']
    modality_index = 0
    for k in range(3):
        pretrain_dir = "../output/models/surf_single_depth_version_"+str(k)+".pth"
        print(pretrain_dir)
        args.gpu = 3
        args.modal = modality[modality_index]
        args.miss_modal = 0
        args.p = moldaity_combination[modality_index]
        args.backbone = "resnet18_se"
        model = resnet18_se(args)
        test_para = torch.load(pretrain_dir)
        model.load_state_dict(torch.load(pretrain_dir))

        result = batch_test_single(model=model, args=args)


if __name__ == '__main__':
    multi_single_test()
