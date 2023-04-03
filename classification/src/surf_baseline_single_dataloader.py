import torchvision.transforms as tt
import torch

from datasets.surf_single_txt import SURF_Single
from lib.processing_utils import get_mean_std

surf_single_transforms_train = tt.Compose(
    [
        tt.Resize((144, 144)),
        tt.RandomRotation(30),
        tt.RandomHorizontalFlip(),
        tt.RandomCrop((112, 112)),
        tt.ToTensor(),
        # tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)



surf_single_transforms_test = tt.Compose(
    [
        # tt.Resize((144, 144)),
        # tt.RandomCrop((112, 112)),
        tt.Resize((112, 112)),
        tt.ToTensor(),
        # tt.Normalize(mean=[0.5, 0.5, 0.5, ], std=[0.5, 0.5, 0.5, ])
    ]
)



def surf_baseline_single_dataloader(train, args):
    # dataset and data loader
    if train:
        txt_dir = args.data_root + '/train_list.txt'
        root_dir = args.data_root

        surf_dataset = SURF_Single(txt_dir=txt_dir,
                                   root_dir=root_dir,
                                   transform=surf_single_transforms_train, modal=args.modal)


    else:
        txt_dir = args.data_root + '/val_private_list.txt'
        root_dir = args.data_root

        surf_dataset = SURF_Single(txt_dir=txt_dir,
                                   root_dir=root_dir,
                                   transform=surf_single_transforms_test, modal=args.modal)

    surf_data_loader = torch.utils.data.DataLoader(
        dataset=surf_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    return surf_data_loader
