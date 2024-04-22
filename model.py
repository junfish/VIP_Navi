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


def model_parser(model, fixed_weight=False, dropout_rate=0.0, bayesian = False):
    base_model = None

    if model == 'Googlenet':
        base_model = models.inception_v3(pretrained=True)
        network = GoogleNet(base_model, fixed_weight, dropout_rate)
    elif model == 'Resnet' or model == 'Resnet34':
        base_model = models.resnet34(pretrained=True)
        network = ResNet34(base_model, fixed_weight, dropout_rate, bayesian)
    elif model == 'Resnet50':
        base_model = models.resnet50(pretrained=True)
        network = ResNet50(base_model, fixed_weight, dropout_rate, bayesian)
    elif model == 'Resnet101':
        base_model = models.resnet101(pretrained=True)  # Load a pretrained ResNet101
        network = ResNet101(base_model, fixed_weight, dropout_rate, bayesian)
    elif model == 'Resnet34Simple':
        base_model = models.resnet34(pretrained=True)
        network = ResNet34Simple(base_model, fixed_weight)
    elif model == 'MobilenetV3':
        base_model = models.mobilenet_v3_large(pretrained=True)  # or models.mobilenet_v3_small(pretrained=True) for the smaller variant
        network = MobileNetV3(base_model, fixed_weight, dropout_rate, bayesian)
    else:
        assert 'Unvalid Model'

    return network


class PoseLoss(nn.Module):
    def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        if not self.learn_beta:
            self.sx = 0
            self.sq = -5
            
        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad=self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad=self.learn_beta)

        # if learn_beta:
        #     self.sx.requires_grad = True
        #     self.sq.requires_grad = True
        #
        # self.sx = self.sx.to(device)
        # self.sq = self.sq.to(device)

        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        pred_q = F.normalize(pred_q, p=2, dim=1)
        # loss_x = F.l1_loss(pred_x, target_x)
        # loss_q = F.l1_loss(pred_q, target_q)

        # loss_x = F.mse_loss(pred_x, target_x)
        # loss_q = F.mse_loss(pred_q, target_q)

        loss_x = torch.mean(F.pairwise_distance(pred_x, target_x, p = 2, eps = 0.0))
        loss_q = torch.mean(F.pairwise_distance(pred_q, target_q, p = 2, eps = 0.0))

            
        loss = torch.exp(-self.sx)*loss_x \
               + self.sx \
               + torch.exp(-self.sq)*loss_q \
               + self.sq

        self.loss_print = [loss.item(), loss_x.item(), loss_q.item()]

        return loss, loss_x.item(), loss_q.item()

class GeometricLoss(nn.Module):
    def __init__(self):
        super(GeometricLoss, self).__init__()

    def __call__(self, *args, **kwargs):
        pass

class ResNet34(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian = False):
        super(ResNet34, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # nn.init.normal_(self.fc_last.weight, 0, 0.01)
        # nn.init.constant_(self.fc_last.bias, 0)
        #
        # nn.init.normal_(self.fc_position.weight, 0, 0.5)
        # nn.init.constant_(self.fc_position.bias, 0)
        #
        # nn.init.normal_(self.fc_rotation.weight, 0, 0.01)
        # nn.init.constant_(self.fc_rotation.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x_fully = self.fc_last(x)
        x = F.relu(x_fully)

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=dropout_on)

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully

class ResNet50(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(ResNet50, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        # Removing the final fully connected layer of the base_model
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)  # No change needed; ResNet50 also ends with 2048 features
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        # Weight initialization remains the same
        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]
        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x_fully = self.fc_last(x)
        x = F.relu(x_fully)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=(self.training or self.bayesian))

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully

class ResNet101(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(ResNet101, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features  # This is 2048 for ResNet101 as well.

        # Removing the final fully connected layer of the base_model
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)  # No change needed
        self.fc_position = nn.Linear(2048, 3, bias=True)
        self.fc_rotation = nn.Linear(2048, 4, bias=True)

        # Weight initialization
        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]
        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x_fully = self.fc_last(x)
        x = F.relu(x_fully)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=(self.training or self.bayesian))

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully


class ResNet34Simple(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0):
        super(ResNet34Simple, self).__init__()
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(feat_in, 3, bias=False)
        self.fc_rotation = nn.Linear(feat_in, 4, bias=False)

        init_modules = [self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight.data)
                if module.bias is not None:
                    nn.init.constant(module.bias.data, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x

class GoogleNet(nn.Module):
    """ PoseNet using Inception V3 """
    def __init__(self, base_model, fixed_weight=False, dropout_rate = 0.0):
        super(GoogleNet, self).__init__()
        self.dropout_rate =dropout_rate

        model = []
        model.append(base_model.Conv2d_1a_3x3)
        model.append(base_model.Conv2d_2a_3x3)
        model.append(base_model.Conv2d_2b_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Conv2d_3b_1x1)
        model.append(base_model.Conv2d_4a_3x3)
        model.append(nn.MaxPool2d(kernel_size=3, stride=2))
        model.append(base_model.Mixed_5b)
        model.append(base_model.Mixed_5c)
        model.append(base_model.Mixed_5d)
        model.append(base_model.Mixed_6a)
        model.append(base_model.Mixed_6b)
        model.append(base_model.Mixed_6c)
        model.append(base_model.Mixed_6d)
        model.append(base_model.Mixed_6e)
        model.append(base_model.Mixed_7a)
        model.append(base_model.Mixed_7b)
        model.append(base_model.Mixed_7c)
        self.base_model = nn.Sequential(*model)

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Out 2
        self.pos2 = nn.Linear(2048, 3, bias=True)
        self.ori2 = nn.Linear(2048, 4, bias=True)

    def forward(self, x):
        # 299 x 299 x 3
        x = self.base_model(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048
        pos = self.pos2(x)
        ori = self.ori2(x)

        return pos, ori, x

class MobileNetV3(nn.Module):
    def __init__(self, base_model, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(MobileNetV3, self).__init__()

        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # MobileNetV3 does not use `fc` but `classifier` for its final layers
        if isinstance(base_model, models.MobileNetV3):
            feat_in = base_model.classifier[0].in_features
            # Modify the base model to not include the final classifier
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        else:
            # Fallback for other model types if necessary
            feat_in = base_model.fc.in_features
            self.base_model = nn.Sequential(*list(base_model.children())[:-1])

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        #### Define the new classifier layers
        self.fc_last = nn.Linear(feat_in, 1280, bias=True)
        self.fc_position = nn.Linear(1280, 3, bias=True)
        self.fc_rotation = nn.Linear(1280, 4, bias=True)
        # self.fc_position = nn.Linear(feat_in, 3, bias=True)
        # self.fc_rotation = nn.Linear(feat_in, 4, bias=True)

        # Initialize weights
        init_modules = [self.fc_last, self.fc_position, self.fc_rotation]
        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)  # Flatten the features
        x_fully = self.fc_last(x)
        x = F.relu(x_fully)

        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=(self.training or self.bayesian))

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully