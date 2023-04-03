import sys

import numpy as np

sys.path.append('..')
from models.surf_baseline import SURF_Baseline, SURF_Multi, SURF_MV, SURF_Baseline_Auxi,SURF_Baseline_Auxi_Weak_Layer4
from src.cefa_baseline_multi_dataloader import cefa_multi_transforms_test,cefa_multi_transforms_train
from lib.model_develop_new import calc_accuracy_multi
from lib.processing_utils import save_csv
from datasets.cefa_multi_protocol import CEFA_Multi
from configuration.cefa_baseline_multi import args
import torch
import torch.nn as nn
import os


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''

    args.data_root = "../data/CeFA-Race"
    cefa_dataset = CEFA_Multi(args=args, mode='test', miss_modal=args.miss_modal, protocol=args.protocol,
                              transform=cefa_multi_transforms_test)
    cefa_data_loader = torch.utils.data.DataLoader(
        dataset=cefa_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4)

    result,_ = calc_accuracy_multi(model=model, loader=cefa_data_loader, verbose=True, hter=True)
    print(result)
    return result



def test_single():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # test_epoch()
    result_list = []
    args.gpu = 1
    args.modal = 'multi'
    args.miss_modal = 0
    args.backbone = "resnet18_se"
    args.inplace_new = 384
    modality=['rgb','depth','ir','rgb_depth','rgb_ir','depth_ir','full']
    index=6

    modality_combination = [[1, 0, 0], [0, 1, 0], [0, 0, 1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]

    for i in range(3):

        pretrain_dir = "../output/models/cefa_"+modality[index]+"_" + str(i) + "__average.pth"

        args.p = modality_combination[index]
        print(args.p)
        model = SURF_Baseline(args)
        test_para = torch.load(pretrain_dir)
        model.load_state_dict(torch.load(pretrain_dir))

        result = batch_test(model=model, args=args)
        result_list.append(result)
        print(result)
    result_arr = np.array((result_list))
    result_mean = np.mean(result_arr, axis=0)
    print(result_mean)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
    # test_epoch()

    test_single()
