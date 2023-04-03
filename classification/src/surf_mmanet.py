import sys

sys.path.append('..')
import torch
from itertools import chain
import os

from src.surf_baseline_multi_dataloader import surf_baseline_multi_dataloader
from cefa_baseline_multi_dataloader import cefa_baseline_multi_dataloader
from models.surf_baseline import SURF_Multi, SURF_MMANet
from loss.kd import *
from lib.model_develop import train_knowledge_distill_patch_feature_auxi_weak
from configuration.config_feature_kd import args
from lib.processing_utils import seed_torch


def deeppix_main(args):

    if args.dataset=='surf':

        train_loader = surf_baseline_multi_dataloader(train=True, args=args)
        test_loader = surf_baseline_multi_dataloader(train=False, args=args)
    elif args.dataset=='cefa':
        train_loader = cefa_baseline_multi_dataloader(train=True, args=args)
        test_loader = cefa_baseline_multi_dataloader(train=False, args=args)
    else:
        raise Exception('error dataset')

    # seed_torch(2)
    print(args)
    args.log_name = args.name + '.csv'
    args.model_name = args.name

    # seed_torch(5)
    teacher_model = SURF_Multi(args)
    student_model = SURF_MMANet(args)

    # 初始化并且固定teacher 网络参数
    if args.dataset=='surf':
        teacher_model.load_state_dict(
            torch.load(os.path.join(args.model_root, 'multi_full__2__average.pth')))
    elif args.dataset=='cefa':
        teacher_model.load_state_dict(
            torch.load(os.path.join(args.model_root, 'cefa_full__1__unaverage.pth')))
    else:
        raise Exception('error dataset')

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False


    # 如果有GPU
    if torch.cuda.is_available():
        teacher_model.cuda()  # 将所有的模型参数移动到GPU上
        student_model.cuda()
        print("GPU is using")

    # define loss functions
    if args.kd_mode == 'sp':
        criterionKD = SP()
    elif args.kd_mode == 'mad':
        criterionKD = MAD()
    else:
        raise Exception('Invalid kd mode...')


    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterionCls = torch.nn.CrossEntropyLoss()

    # initialize optimizer

    if args.optim == 'sgd':
        print('--------------------------------optim with sgd--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.SGD(chain(student_model.parameters(),
                                              *[c.parameters() for c in criterionKD[1:]]),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
        else:
            optimizer = torch.optim.SGD(student_model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay,
                                        nesterov=True)
    elif args.optim == 'adam':
        print('--------------------------------optim with adam--------------------------------------')
        if args.kd_mode in ['vid', 'ofd', 'afd']:
            optimizer = torch.optim.Adam(chain(student_model.parameters(),
                                               *[c.parameters() for c in criterionKD[1:]]),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
        else:
            optimizer = torch.optim.Adam(student_model.parameters(),
                                         lr=args.lr,
                                         weight_decay=args.weight_decay,
                                         )
    else:
        print('optim error')
        optimizer = None

    # warp nets and criterions for train and test
    nets = {'snet': student_model, 'tnet': teacher_model}
    criterions = {'criterionCls': criterionCls, 'criterionKD': criterionKD}

    train_knowledge_distill_patch_feature_auxi_weak(net_dict=nets, cost_dict=criterions, optimizer=optimizer,
                                                    train_loader=train_loader,
                                                    test_loader=test_loader,
                                                    args=args)


if __name__ == '__main__':
    deeppix_main(args)
