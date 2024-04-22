import os
import time
import copy
import torch
import torchvision
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from colmap.scripts.python.read_write_model import read_model
import pdb


class CustomDataset(Dataset):
    def __init__(self, proj_path, metadata_path, mode, transform, num_val=100):
        self.proj_path = proj_path
        self.metadata_path = metadata_path
        self.mode = mode
        self.transform = transform
        raw_lines = open(self.metadata_path, 'r').readlines()
        self.lines = raw_lines[2:]

        # print(self.lines.__len__())
        print("An example of a single line of %s data appears: " % self.mode)
        print(self.lines[1])

        self.test_filenames = []
        self.test_poses = []
        self.train_filenames = []
        self.train_poses = []

        for i, line in enumerate(self.lines):
            splits = line.split()
            filename = splits[1].replace(",", "")
            values = splits[2:]
            values = list(map(lambda x: float(x.replace(",", "")), values))
            filename = os.path.join(self.proj_path, filename)

            if self.mode == 'train':
                # if i > num_val:
                self.train_filenames.append(filename)
                self.train_poses.append(values)
            elif self.mode == 'test':
                self.test_filenames.append(filename)
                self.test_poses.append(values)
            elif self.mode == 'val':
                self.test_filenames.append(filename)
                self.test_poses.append(values)
                if i > num_val:
                    break
            else:
                assert 'Unavailable mode'

        self.num_train = self.train_filenames.__len__()
        self.num_test = self.test_filenames.__len__()
        print("Number of Train: ", self.num_train)
        print("Number of Test: ", self.num_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(self.train_filenames[index])
            pose = self.train_poses[index]
        elif self.mode in ['val', 'test']:
            image = Image.open(self.test_filenames[index])
            pose = self.test_poses[index]

        return self.transform(image), torch.Tensor(pose)

    def __len__(self):
        if self.mode == 'train':
            num_data = self.num_train
        elif self.mode in ['val', 'test']:
            num_data = self.num_test
        return num_data

class COLMAPDataset(Dataset):
    def __init__(self, proj_path, metadata_path, mode, transform, num_val=100):
        self.proj_path = proj_path
        self.metadata_path = metadata_path
        self.mode = mode
        self.transform = transform
        raw_lines = open(self.metadata_path, 'r').readlines()
        self.lines = raw_lines[2:]


        # print(self.lines.__len__())
        # print("An example of a single line of %s data appears: " % self.mode)
        # print(self.lines[1])

        # self.test_filenames = []
        # self.test_poses = []
        # self.train_filenames = []
        # self.train_poses = []
        self.train_data = []
        self.test_data = []
        self.train_global_depths = []
        self.test_global_depths = []

        for i, line in enumerate(self.lines):
            splits = line.split()
            filename = splits[1].replace(",", "")
            values = splits[2:]
            values = list(map(lambda x: float(x.replace(",", "")), values))
            filename = os.path.join(self.proj_path, filename)

            if self.mode == 'train':
                # if i > num_val:
                self.train_filenames.append(filename)
                self.train_poses.append(values)
            elif self.mode == 'test':
                self.test_filenames.append(filename)
                self.test_poses.append(values)
            elif self.mode == 'val':
                self.test_filenames.append(filename)
                self.test_poses.append(values)
                if i > num_val:
                    break
            else:
                assert 'Unavailable mode'

        self.num_train = self.train_filenames.__len__()
        self.num_test = self.test_filenames.__len__()
        print("Number of Train: ", self.num_train)
        print("Number of Test: ", self.num_test)
    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(self.train_filenames[index])
            data = self.train_data[index]
        elif self.mode in ['val', 'test']:
            image = Image.open(self.test_filenames[index])
            pose = self.test_data[index]

        return self.transform(image), torch.Tensor(pose)

    def __len__(self):
        if self.mode == 'train':
            num_data = self.num_train
        elif self.mode in ['val', 'test']:
            num_data = self.num_test
        return num_data



def get_loader(model, proj_path, metadata_path, mode, batch_size, is_shuffle=False, num_val=100):

    # Predefine image size
    if model == 'Googlenet':
        img_size = 300
        img_crop = 299
    elif model in ['Resnet', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet34Simple', 'MobilenetV3']:
        img_size = 256
        img_crop = 224


    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_crop),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # metadata_path_val = '/mnt/data2/image_based_localization/posenet/KingsCollege/dataset_test.txt'
        datasets = {'train': CustomDataset(proj_path, metadata_path, 'train', transform, num_val),
                    'val': CustomDataset(proj_path, metadata_path, 'val', transform, num_val)}
        # data_loaders = {x: DataLoader(datasets[x], batch_size, is_shuffle, num_workers=batch_size)
        #                 for x in ['train', 'val']}
        data_loaders = {'train': DataLoader(datasets['train'], batch_size, is_shuffle, num_workers=4),
                        'val': DataLoader(datasets['val'], batch_size, is_shuffle, num_workers=4)}
    elif mode == 'test':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        batch_size = 1
        is_shuffle = False
        dataset = CustomDataset(proj_path, metadata_path, 'test', transform)
        data_loaders = DataLoader(dataset, batch_size, is_shuffle, num_workers=4)

    else:
        assert 'Unavailable Mode'

    return data_loaders

cameras, images, point3D = read_model('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/Floor_3/20240215_203732_proj/sparse/geo')
img2id = {image.name: i for i, image in images.items()}
scene_coordinates = torch.zeros(max(point3D.keys()) + 1, 3, dtype = torch.float64)
for i, point3D in point3D.items():
    scene_coordinates[i] = torch.tensor(point3D.xyz)
pass
print('!')