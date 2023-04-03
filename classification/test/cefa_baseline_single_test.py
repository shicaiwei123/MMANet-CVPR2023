import sys

sys.path.append('..')
from models.resnet18_se import resnet18_se
from src.cefa_baseline_single_dataloader import cefa_single_transforms_test
from lib.model_develop_utils import calc_accuracy
from datasets.cefa_single_protocol import CEFA_Single
from configuration.config_baseline_single import args
import torch
import torch.nn as nn
import torchvision.models as tm


def batch_test(model, args):
    '''
    利用dataloader 装载测试数据,批次进行测试
    :return:
    '''
    args.data_root = "/home/data/shicaiwei/cefa/CeFA-Race"
    cefa_dataset = CEFA_Single(args=args, modal=args.modal, mode='test', protocol=args.protocol,
                               transform=cefa_single_transforms_test)
    cefa_data_loader = torch.utils.data.DataLoader(
        dataset=cefa_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4)

    result = calc_accuracy(model=model, loader=cefa_data_loader, verbose=True, hter=True)
    print(result)


if __name__ == '__main__':
    for i in range(5):
        pretrain_dir = "../output/models/cefa_resnet18_se_dropout_no_seed_profile_version_" + str(1) + ".pth"
        args.modal = pretrain_dir.split('_')[-3]
        args.gpu = 3
        args.backbone = "resnet18_se"
        args.version = 0
        model = resnet18_se(args, pretrained=False)
        model.load_state_dict(torch.load(pretrain_dir))

        batch_test(model=model, args=args)
