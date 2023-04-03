import sys

sys.path.append('..')
from models.surf_baseline import SURF_Multi,SURF_Baseline
from models.surf_baseline import SURF_MV,SURF_Baseline
from src.cefa_baseline_multi_dataloader import cefa_multi_transforms_train, cefa_multi_transforms_test
from lib.model_develop_new import calc_accuracy_multi
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
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    result = calc_accuracy_multi(model=model, loader=cefa_data_loader, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(3)

    # pretrain_dir = "../output/models/cefa_multi_2_.pth"
    # args.gpu = 0
    # args.miss_modal = 0
    # args.backbone = "resnet18_se"
    # args.modal='multi'
    # model = SURF_Multi(args)
    # print(pretrain_dir)
    # model.load_state_dict(torch.load(pretrain_dir,map_location='cpu'))
    #
    # model=model.cuda()

    pretrain_dir = "../output/models/cefa_full__1__unaverage.pth"
    args.gpu = 0
    args.backbone = "resnet18_se"
    args.p = [1, 1, 1]
    args.inplace_new=384
    model = SURF_Baseline(args)
    print(pretrain_dir)
    model.load_state_dict(torch.load(pretrain_dir, map_location='cpu'))
    model.eval()

    model = model.cuda()

    batch_test(model=model, args=args)
