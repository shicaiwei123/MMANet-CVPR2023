from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as ttf
import numpy as np

from datasets.cefa_dataset_class import *


class CEFA_Single(Dataset):
    def __init__(self, args, modal, mode, protocol, transform=None):
        self.args = args
        cefa_dataset = load_casia_race(args.data_root, modal=modal, protocol=protocol,
                                       mode=mode)
        image_list, label_list = get_sframe_paths_labels(cefa_dataset, mode, ratio=1)
        print(len(label_list), sum(np.array(label_list)))

        # # 以相同的方式打乱顺序
        # dataset = list(zip(image_list, label_list))
        # random.shuffle(dataset)
        # self.image_list, self.label_list = zip(*dataset)
        self.image_list = image_list
        self.label_list = label_list

        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        label = self.label_list[idx]
        img_pil = Image.open(image_path)
        if self.transform is not None:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = ttf.to_tensor(img_pil)

        return img_tensor, label

    def __len__(self):
        return len(self.label_list)
