"""
Function: Class CASIA_Race
Format: Race/Race-ID-IdCard/Race_ID_AcqDevice_Session_PAI
    Race: 1:AF 2:CA 3:EA
    ID: 000-599
    AcqDevice: 1:rssdk 2:mp4  3:bag 4:MOV
    Session(environment): 1:indoor 2:outdoor 3:random
    PAI: 1:Real 2:Cloth 3:Pic(phtoto) 4:Screen
Example: EA/EA-012-198006250024/1_000_1_1_1(P1_P2_P3_P4_P5)
Info:
AF: Num = 2,094,730, RAM = 260G
CA: Num = 2,025,816, RAM = 269G
EA: Num = 1,884,592, RAM = 244G

Author: AJ
Date: 2019.9.27
"""

import os, random
from datasets.cefa_protocol_class import *


class ImageClass():
    """
    Stores the paths of images for a given video
    input: video_name, image_paths
    output: class(include three functions)
    """

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


def load_casia_race(data_path, protocol, mode, modal='profile', is_sort=True):
    FILES_LIST = []
    dataset = []
    All_Race = ['AF', 'CA', 'EA']
    assert All_Race[0] == 'AF' and All_Race[1] == 'CA' and All_Race[2] == 'EA'
    for race in All_Race:
        All_Sunjects = os.listdir(os.path.join(data_path, race))
        All_Sunjects.sort()
        for subject in All_Sunjects:
            All_Videos = os.listdir(os.path.join(data_path, race, subject))
            All_Videos.sort()
            # print(All_Videos)
            for video in All_Videos:
                FILES_LIST.append(video)
    if 'rdi' in protocol:
        casia_race = CASIA_Race_RDI(protocol, mode)
    else:
        casia_race = ''

    # FILES_LIST.sort()
    FILES_LIST = casia_race.dataset_process(FILES_LIST)
    for i in range(len(FILES_LIST)):
        video = FILES_LIST[i]
        # print(video)
        P = video.split('_')
        race = All_Race[int(P[0]) - 1]
        race_id = '-'.join([race, P[1]])
        profiledir = os.path.join(data_path, race, race_id, video, modal)
        All_Images = os.listdir(profiledir)
        All_Images.sort()
        assert len(All_Images) >= 10
        image_paths = [os.path.join(profiledir, img) for img in All_Images]
        if is_sort:
            image_paths.sort()  ### Guaranteed continuous frames
        else:
            random.shuffle(image_paths)  ### Shuffle continuous frames

        # print(image_paths)
        dataset.append(ImageClass(video, image_paths))

    random.shuffle(dataset)
    return dataset


def load_casia_mask(data_path, protocol, mode, modal='profile', is_sort=True):
    FILES_LIST = []
    dataset = []
    All_Mask = ['3D-Mask', 'Silicone-Mask']
    assert (All_Mask[0] == '3D-Mask') and (All_Mask[1] == 'Silicone-Mask')
    for mask in All_Mask:
        All_Sunjects = os.listdir(os.path.join(data_path, mask))
        for subject in All_Sunjects:
            All_Videos = os.listdir(os.path.join(data_path, mask, subject))
            for video in All_Videos:
                FILES_LIST.append(video)
    if 'rdi' in protocol:
        casia_mask = CASIA_Mask_RDI(protocol, mode)
    else:
        pass
    FILES_LIST = casia_mask.dataset_process(FILES_LIST)
    for i in range(len(FILES_LIST)):
        video = FILES_LIST[i]
        P = video.split('_')
        mask = All_Mask[int(P[0]) - 1]
        profiledir = os.path.join(data_path, mask, P[1], video, modal)
        All_Images = os.listdir(profiledir)
        assert len(All_Images) >= 1
        image_paths = [os.path.join(profiledir, img) for img in All_Images]
        if is_sort:
            image_paths.sort()  ### Guaranteed continuous frames
        else:
            random.shuffle(image_paths)  ### Shuffle continuous frames
        dataset.append(ImageClass(video, image_paths))
    if is_sort:
        dataset.sort()
    else:
        random.shuffle(dataset)
    return dataset


### First(training): real(label=0), attack(label=1) ###
def video_2_label(video_name):
    label = int(video_name.split('_')[-1])
    label = 0 if label == 1 else 1
    return label


def get_sframe_paths_labels(dataset, phase, num='all', ratio=1):
    image_paths_flat = []
    labels_flat = []
    for i in range(len(dataset)):
        label = video_2_label(dataset[i].name)
        if phase == 'train':
            if label == 0:
                ratio_ = 1  ### real
            else:
                ratio_ = ratio  ### fake
            sample_image_paths = \
                [dataset[i].image_paths[sam_idx] for sam_idx in range(0, len(dataset[i].image_paths), ratio_)]


            sample_image_paths.sort()
            # print(sample_image_paths)
            image_paths_flat += sample_image_paths
            labels_flat += [label] * len(sample_image_paths)
        elif (phase == 'dev') or (phase == 'test'):
            if num == 'one':
                load_num = 1
            elif num == 'all':
                load_num = len(dataset[i].image_paths)
            ### image_paths_flat += random.sample(dataset[i].image_paths, batch_size_val)
            image_paths_flat += dataset[i].image_paths[0:0 + load_num]  ### In order to get stable results
            labels_flat += [label] * load_num
    assert len(image_paths_flat) == len(labels_flat)
    return image_paths_flat, labels_flat
