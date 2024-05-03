import os
import time
import copy
import torch
import glob
import tqdm
import cv2
import pickle
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
from dataset_utils.world_coordinates import quaternion_R_matrix
from kornia.geometry.conversions import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix

train_test_split_dict = {
    "Basement": 60,
    "Lower_Level": 90,
    "Floor_2": 75,
    "Floor_3": 75
}

def geo_collate_fn(views):
    """
    Tranforms list of dicts [{key1: value1, key2: value2}, {key1: value3, key2: value4}]
    into a dict of lists {key1: {value1, value3}, key2: {value2, value4}}
    """
    batch = {key: [] for key in views[0].keys()}
    for view in views:
        for key, value in view.items():
            batch[key].append(value)
    for key, value in batch.items():
        if key not in ['w_P', 'c_p']:
            batch[key] = torch.stack(value)
    return batch

class CustomDataset(Dataset):
    def __init__(self, proj_path, metadata_path, mode, transform, num_val=100):
        self.proj_path = proj_path
        self.metadata_path = metadata_path
        self.mode = mode
        self.transform = transform
        raw_lines = open(self.metadata_path, 'r').readlines()
        self.lines = raw_lines[2:]

        # print(self.lines.__len__())
        print("An example of a single line of %s data appears (before shuffling): " % self.mode)
        print(self.lines[0])

        self.test_filenames = []
        self.test_poses = []
        self.train_filenames = []
        self.train_poses = []

        for i, line in enumerate(self.lines):
            splits = line.split()
            filename = splits[0].replace(",", "")
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
        # print("Number of Train: ", self.num_train)
        # print("Number of Test: ", self.num_test)

    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(self.train_filenames[index])
            pose = self.train_poses[index]
        elif self.mode in ['val', 'test']:
            image = Image.open(self.test_filenames[index])
            pose = self.test_poses[index]

        return {"image": self.transform(image), "pose": torch.Tensor(pose)}

    def __len__(self):
        if self.mode == 'train':
            num_data = self.num_train
        elif self.mode in ['val', 'test']:
            num_data = self.num_test
        return num_data

class GeometricDataset(Dataset):
    def __init__(self, proj_path, metadata_path, mode, transform, num_val=100):
        self.proj_path = proj_path
        self.metadata_path = metadata_path
        self.mode = mode
        try:
            with open(self.metadata_path, 'rb') as file:
                data = pickle.load(file)
        except FileNotFoundError:
            print("The file was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        self.train_data, self.test_data = [], []
        if self.mode == "train":
            self.train_data = data['train']
        elif self.mode == "test":
            self.test_data = data['test']
        elif self.mode == "val":
            self.test_data = data['train'][:num_val]


        self.transform = transform

        # 'image_path': os.path.join(colmap_proj_name, image_folder, image.name),
        # 'w_t_c': w_t_c.float(),
        # 'c_q_w': c_q_w.float(),
        # 'c_R_w': c_R_w.float(),
        # 'K': new_K.float(),
        # 'w_P': w_P.float(),
        # 'c_p': c_p.T.float(),

        self.num_train = self.train_data.__len__()
        self.num_test = self.test_data.__len__()
        print("Number of Train: ", self.num_train)
        print("Number of Test: ", self.num_test)
    def __getitem__(self, index):
        if self.mode == 'train':
            image = Image.open(os.path.join(self.proj_path, self.train_data[index]['image_path']))
            # data_point = self.train_data[index]
            # if isinstance(data_point, dict):
            #     data_patch =  {key: value.clone() for key, value in data_point.items()}
            # data_patch['image'] = self.transform(image)
            data_patch = {
                'image': self.transform(image),
                'w_t_c': self.train_data[index]['w_t_c'].clone(),
                'c_q_w': self.train_data[index]['c_q_w'].clone(),
                'c_R_w': self.train_data[index]['c_R_w'].clone(),
                'w_P': self.train_data[index]['w_P'].clone(),
                'c_p': self.train_data[index]['c_p'].clone(),
                'K': self.train_data[index]['K'].clone(),
            }
        elif self.mode in ['val', 'test']:
            image = Image.open(os.path.join(self.proj_path, self.test_data[index]['image_path']))
            # data_point = self.train_data[index]
            # if isinstance(data_point, dict):
            #     data_patch = {key: value.clone() for key, value in data_point.items()}
            # data_patch['image'] = self.transform(image)
            data_patch = {
                'image': self.transform(image),
                'w_t_c': self.test_data[index]['w_t_c'].clone(),
                'c_q_w': self.test_data[index]['c_q_w'].clone(),
                'c_R_w': self.test_data[index]['c_R_w'].clone(),
                'w_P': self.test_data[index]['w_P'].clone(),
                'c_p': self.test_data[index]['c_p'].clone(),
                'K': self.test_data[index]['K'].clone(),
            }

        return data_patch

    def __len__(self):
        if self.mode == 'train':
            num_data = self.num_train
        elif self.mode in ['val', 'test']:
            num_data = self.num_test
        return num_data


def get_loader(model, proj_path, metadata_path, mode, geometric, batch_size, is_shuffle=False, num_val=100):

    # Predefine image size
    if model == 'Googlenet':
        img_size = 300
        img_crop = 299
    elif model in ['Resnet', 'Resnet34', 'Resnet50', 'Resnet101', 'Resnet34Simple', 'MobilenetV3', 'Resnet34lstm', 'MobilenetV3lstm', 'Resnet34hourglass', "MobilenetV3hourglass"]:
        img_size = 256
        img_crop = 224

    # if geometric:
    #     transform = transforms.Compose([
    #         transforms.Resize(img_size),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #     ])

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomCrop(img_crop),
            transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if geometric:
            datasets = {'train': GeometricDataset(proj_path, metadata_path, 'train', transform, num_val),
                        'val': GeometricDataset(proj_path, metadata_path, 'val', transform, num_val)}
            # data_loaders = {x: DataLoader(datasets[x], batch_size, is_shuffle, num_workers=batch_size)
            #                 for x in ['train', 'val']}
            data_loaders = {'train': DataLoader(datasets['train'], batch_size, is_shuffle, collate_fn = geo_collate_fn, pin_memory = True, num_workers=4, drop_last = True),
                            'val': DataLoader(datasets['val'], batch_size, is_shuffle, collate_fn = geo_collate_fn, pin_memory = True, num_workers=4, drop_last = True)}
        else:
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
        if geometric:
            dataset = GeometricDataset(proj_path, metadata_path, 'test', transform, num_val)
            data_loaders = DataLoader(dataset, batch_size, is_shuffle, collate_fn = geo_collate_fn, pin_memory = True, num_workers=4, drop_last = True)
        else:
            dataset = CustomDataset(proj_path, metadata_path, 'test', transform)
            data_loaders = DataLoader(dataset, batch_size, is_shuffle, num_workers=4)

    else:
        assert 'Unavailable Mode'

    return data_loaders
