from datasets.surf_txt import Resize_multi, Normaliztion_multi, ToTensor_multi, RandomHorizontalFlip_multi, \
    RondomCrop_multi, RondomRotion_multi, Cutout_multi
from datasets.cefa_multi_protocol import CEFA_Multi

import torchvision.transforms as tt
import torch

cefa_multi_transforms_train = tt.Compose(
    [
        Resize_multi((144, 144)),
        RondomRotion_multi(30),
        RondomCrop_multi(112),
        RandomHorizontalFlip_multi(),
        ToTensor_multi(),
        # Cutout_multi(20),
        Normaliztion_multi(),

    ]
)

cefa_multi_transforms_test = tt.Compose(
    [
        # Resize_multi((144, 144)),
        # RondomCrop_multi(112),
        Resize_multi((112, 112)),
        # RandomHorizontalFlip_multi(),
        ToTensor_multi(),
        Normaliztion_multi(),
    ]
)


def cefa_baseline_multi_dataloader(train, args):
    # dataset and data loader
    if train:

        cefa_dataset = CEFA_Multi(args=args, mode='train', protocol=args.protocol,
                                  transform=cefa_multi_transforms_train, miss_modal=args.miss_modal)

    else:

        cefa_dataset = CEFA_Multi(args=args, mode='dev', protocol=args.protocol,
                                  transform=cefa_multi_transforms_test, miss_modal=args.miss_modal)

    cefa_data_loader = torch.utils.data.DataLoader(
        dataset=cefa_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    return cefa_data_loader
