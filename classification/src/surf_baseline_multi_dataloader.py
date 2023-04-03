from datasets.surf_txt import Resize_multi, Normaliztion_multi, ToTensor_multi, RandomHorizontalFlip_multi, \
    RondomCrop_multi, RondomRotion_multi, Cutout_multi
from datasets.surf_txt import SURF

import torchvision.transforms as tt
import torch

surf_multi_transforms_train = tt.Compose(
    [
        Resize_multi((144, 144)),
        RondomRotion_multi(30),
        RondomCrop_multi(112),
        RandomHorizontalFlip_multi(),
        ToTensor_multi(),
        # Cutout_multi(30),
        Normaliztion_multi(),

    ]
)

surf_multi_transforms_test = tt.Compose(
    [
        # Resize_multi((144, 144)),
        # RondomCrop_multi(112),
        Resize_multi((112, 112)),
        # RandomHorizontalFlip_multi(),
        ToTensor_multi(),
        # Cutout_multi(30),
        Normaliztion_multi(),
    ]
)


def surf_baseline_multi_dataloader(train, args):
    # dataset and data loader
    if train:
        txt_dir = args.data_root + '/train_list.txt'
        root_dir = args.data_root
        # txt_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training.txt"
        # root_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR"
        surf_dataset = SURF(txt_dir=txt_dir,
                            root_dir=root_dir,
                            transform=surf_multi_transforms_train, miss_modal=args.miss_modal)

        surf_data_loader = torch.utils.data.DataLoader(
            dataset=surf_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=True
        )

    else:
        txt_dir = args.data_root + '/val_private_list.txt'
        root_dir = args.data_root
        # txt_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR/Training.txt"
        # root_dir="/home/shicaiwei/data/liveness_data/CASIA-SUFR"

        surf_dataset = SURF(txt_dir=txt_dir,
                            root_dir=root_dir,
                            transform=surf_multi_transforms_test, miss_modal=args.miss_modal, times=1)

        surf_data_loader = torch.utils.data.DataLoader(
            dataset=surf_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            drop_last = True

        )

    return surf_data_loader
