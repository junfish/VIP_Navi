import os
import pdb
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
from torchvision.models import (Inception_V3_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights)
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from kornia.geometry.conversions import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix


def model_parser(model, use_euler6 = False, fixed_weight = False, dropout_rate = 0.0, bayesian = False):

    base_model = None
    weights = None

    # Assigning the correct pretrained weight enums
    if model == 'Googlenet':
        weights = Inception_V3_Weights.IMAGENET1K_V1
        base_model = models.inception_v3(weights = weights)
        network = GoogleNet(base_model, use_euler6, fixed_weight, dropout_rate)

    elif model == 'Resnet' or model == 'Resnet34':
        weights = ResNet34_Weights.IMAGENET1K_V1
        base_model = models.resnet34(weights = weights)
        network = ResNet34(base_model, use_euler6, fixed_weight, dropout_rate, bayesian)

    elif model == 'Resnet34Simple':
        weights = ResNet34_Weights.IMAGENET1K_V1
        base_model = models.resnet34(weights = weights)
        network = ResNet34Simple(base_model, use_euler6, fixed_weight)

    elif model == 'Resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1
        base_model = models.resnet50(weights = weights)
        network = ResNet50(base_model, use_euler6, fixed_weight, dropout_rate, bayesian)

    elif model == 'Resnet101':
        weights = ResNet101_Weights.IMAGENET1K_V1
        base_model = models.resnet101(weights = weights)
        network = ResNet101(base_model, use_euler6, fixed_weight, dropout_rate, bayesian)

    elif model == 'MobilenetV3':
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        base_model = models.mobilenet_v3_large(weights = weights)
        network = MobileNetV3(base_model, use_euler6, fixed_weight, dropout_rate, bayesian)

    elif model == 'Resnet34lstm':
        weights = ResNet34_Weights.IMAGENET1K_V1
        base_model = models.resnet34(weights = weights)
        network = ResNet34LSTM(base_model, use_euler6, fixed_weight, 256, dropout_rate, bayesian)

    elif model == 'MobilenetV3lstm':
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        base_model = models.mobilenet_v3_large(weights=weights)
        network = MobileNetV3LSTM(base_model, use_euler6, fixed_weight, 256, dropout_rate, bayesian)

    elif model == 'Resnet34hourglass':
        weights = ResNet34_Weights.IMAGENET1K_V1
        base_model = models.resnet34(weights = weights)
        network = ResNet34Hourglass(base_model, use_euler6, fixed_weight, False, dropout_rate, bayesian)

    elif model == "MobilenetV3hourglass":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        base_model = models.mobilenet_v3_large(weights = weights)
        network = MobileNetV3Hourglass(base_model, use_euler6, fixed_weight, True, dropout_rate, bayesian)

    elif model == "Branchresnet34":
        weights = ResNet34_Weights.IMAGENET1K_V1
        base_model = models.resnet34(weights=weights)
        network = BranchResNet34(base_model, use_euler6, fixed_weight, dropout_rate, bayesian)

    elif model == "BranchmobilenetV3":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
        base_model = models.mobilenet_v3_large(weights=weights)
        network = BranchMobileNetV3(base_model, use_euler6, fixed_weight, dropout_rate, bayesian)

    else:
        assert False, 'Invalid Model'

    return network


class PoseLoss(nn.Module):
    def __init__(self, device, sx=0.0, sq=0.0, learn_beta=False):
        super(PoseLoss, self).__init__()
        self.learn_beta = learn_beta

        # if not self.learn_beta:
        #     self.sx = 0
        #     self.sq = -5
            
        self.sx = nn.Parameter(torch.Tensor([sx]), requires_grad = self.learn_beta)
        self.sq = nn.Parameter(torch.Tensor([sq]), requires_grad = self.learn_beta)

        # self.sx = self.sx.to(device)
        # self.sq = self.sq.to(device)

        # if learn_beta:
        #     self.sx.requires_grad = True
        #     self.sq.requires_grad = True
        #
        # self.sx = self.sx.to(device)
        # self.sq = self.sq.to(device)

        self.loss_print = None

    def forward(self, pred_x, pred_q, target_x, target_q):
        '''
        It doesn't matter whether q is a quaternion or in euler6 format.
        '''
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
    # Kendall, Alex, and Roberto Cipolla. "Geometric loss functions for camera pose regression with deep learning." In Proceedings of the IEEE CVPR, pp. 5974-5983. 2017.

    def __init__(self, device):
        super(GeometricLoss, self).__init__()
        self.loss_print = None
        self.device = device

    def forward(self, pred_x, pred_q, target_x, target_q, c_R_w, w_P):
        '''
        :param pred_x: (batch_size, 3)
        :param pred_q: (batch_size, 4)
        :param target_x: (batch_size, 3, 1)
        :param target_q: (batch_size, 4)
        '''
        # It is prohibited using euler6 in Geometric Loss

        pred_q = F.normalize(pred_q, p=2, dim=1)

        pred_q[:, 1:] *= -1 # back to q_hat in Eq.(6)
        pred_R = quaternion_to_rotation_matrix(pred_q) # (batch_size, 3, 3)

        loss_x = torch.mean(F.pairwise_distance(pred_x, target_x.squeeze(), p=2, eps=0.0))
        loss_q = torch.mean(F.pairwise_distance(pred_q, target_q, p=2, eps=0.0))

        pred_x = pred_x.view(-1, 3, 1) # resize the shape from (batch_size, 3) to (batch_size, 3, 1); x_hat in Eq.(6)

        # batch_error = torch.tensor(0.0, device=self.device)
        batch_error = 0.0
        for pred_x_item, pred_R_item,  w_t_c_item, c_R_w_item, w_P_item in zip(pred_x, pred_R, target_x, c_R_w, w_P):
            target_c_p = c_R_w_item @ (w_P_item.T - w_t_c_item)
            pred_c_p = pred_R_item @ (w_P_item.T - pred_x_item) # (3, 3) @ (3, |G|) --> (3, |G|)
            # pred_c_p = K @ pred_c_P  # K is set to identity in Eq.(6)
            target_c_p = target_c_p[:2] / target_c_p[2]
            pred_c_p = pred_c_p[:2] / pred_c_p[2] # pred_c_p: (2, |G|) (u, v, w) --> (u, v)
            sample_error = torch.mean(F.pairwise_distance(pred_c_p.T, target_c_p.T, p=1, eps=0.0))  # .clip(0,100) # c_p_item: (|G|, 2)
            if torch.isnan(sample_error):
                sample_error = sample_error.nan_to_num()
            batch_error += sample_error

        batch_error /= pred_q.shape[0]

        self.loss_print = [batch_error.item(), loss_x.item(), loss_q.item()]

        return batch_error, loss_x.item(), loss_q.item()

class ResNet34(nn.Module):
    def __init__(self, base_model, use_euler6, fixed_weight=False, dropout_rate=0.0, bayesian = False):
        super(ResNet34, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
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
        self.fc_rotation = nn.Linear(2048, self.ori_dim, bias=True)

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
    def __init__(self, base_model, use_euler6, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(ResNet50, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
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
        self.fc_rotation = nn.Linear(2048, self.ori_dim, bias=True)

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
    def __init__(self, base_model, use_euler6, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(ResNet101, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
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
        self.fc_rotation = nn.Linear(2048, self.ori_dim, bias=True)

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
    def __init__(self, base_model, use_euler6, fixed_weight=False):
        super(ResNet34Simple, self).__init__()

        self.ori_dim = 6 if use_euler6 else 4
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(feat_in, 3, bias=False)
        self.fc_rotation = nn.Linear(feat_in, self.ori_dim, bias=False)

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
    def __init__(self, base_model, use_euler6, fixed_weight=False, dropout_rate = 0.0):
        super(GoogleNet, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
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
        # Out 1

        # Out 2
        self.fc_position_2 = nn.Linear(2048, 3, bias=True)
        self.fc_rotation_2 = nn.Linear(2048, self.ori_dim, bias=True)

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
        pos = self.fc_position_2(x)
        ori = self.fc_rotation_2(x)

        return pos, ori, x

class MobileNetV3(nn.Module):
    def __init__(self, base_model, use_euler6, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(MobileNetV3, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
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
        self.fc_rotation = nn.Linear(1280, self.ori_dim, bias=True)
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


class ResNet34LSTM(nn.Module): # For streamlined coding style, which can be combined with ResNet34
    def __init__(self, base_model, use_euler6, fixed_weight = False, hidden_size = 256, dropout_rate = 0.0, bayesian = False):
        super(ResNet34LSTM, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.lstm_l2r = nn.LSTM(input_size = 64, hidden_size = hidden_size, bidirectional=True, batch_first=True)
        self.lstm_u2d = nn.LSTM(input_size = 32, hidden_size = hidden_size, bidirectional=True, batch_first=True)
        self.fc_position = nn.Linear(hidden_size * 4, 3, bias=True) # in_features: hidden_size * 4
        self.fc_rotation = nn.Linear(hidden_size * 4, self.ori_dim, bias=True)

        init_modules = [self.fc_last, self.lstm_l2r, self.lstm_u2d, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'bias' in name: # bias_ih_l; bias_hh_l
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name: # weight_ih_l; weight_hh_l
                        nn.init.xavier_normal_(param)

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

        x = x.view(x.size(0), 32, -1) # --> (batch_size, 32, 64)
        _, (hidden_state_l2r, _) = self.lstm_l2r(x.permute(0, 1, 2)) # (2, batch_size, hidded_size)
        _, (hidden_state_u2d, _) = self.lstm_u2d(x.permute(0, 2, 1)) # (2, batch_size, hidded_size)

        x = torch.cat((hidden_state_l2r[0, :, :],
                            hidden_state_l2r[1, :, :],
                            hidden_state_u2d[0, :, :],
                            hidden_state_u2d[1, :, :]), 1)

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training = dropout_on)

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully

class MobileNetV3LSTM(nn.Module):
    def __init__(self, base_model, use_euler6, fixed_weight=False, hidden_size = 256, dropout_rate=0.0, bayesian=False):
        super(MobileNetV3LSTM, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
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

        # Replace the last layer
        self.fc_last = nn.Linear(feat_in, 1280, bias=True)
        self.lstm_l2r = nn.LSTM(input_size = 40, hidden_size = hidden_size, bidirectional=True, batch_first=True)
        self.lstm_u2d = nn.LSTM(input_size = 32, hidden_size = hidden_size, bidirectional=True, batch_first=True)
        self.fc_position = nn.Linear(hidden_size * 4, 3, bias=True)
        self.fc_rotation = nn.Linear(hidden_size * 4, self.ori_dim, bias=True)

        init_modules = [self.fc_last, self.lstm_l2r, self.lstm_u2d, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'bias' in name:
                        nn.init.constant_(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal_(param)

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)  # Flatten the output for the FC layer
        x_fully = self.fc_last(x)
        x = F.relu(x_fully)

        x = x.view(x.size(0), 32, -1)  # --> (batch_size, 32, 40)
        _, (hidden_state_l2r, _) = self.lstm_l2r(x.permute(0, 1, 2))  # (2, batch_size, hidded_size)
        _, (hidden_state_u2d, _) = self.lstm_u2d(x.permute(0, 2, 1))  # (2, batch_size, hidded_size)

        x = torch.cat((hidden_state_l2r[0, :, :],
                       hidden_state_l2r[1, :, :],
                       hidden_state_u2d[0, :, :],
                       hidden_state_u2d[1, :, :]), 1)

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=dropout_on)

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully

class ResNet34Hourglass(nn.Module):
    def __init__(self, base_model, use_euler6, fixed_weight = False, sum_mode = True, dropout_rate = 0.0, bayesian = False):
        super(ResNet34Hourglass, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
        self.sum_mode = sum_mode
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        if fixed_weight:
            for param in base_model.parameters():
                param.requires_grad = False
        # Encoding Blocks
        self.init_block = nn.Sequential(*list(base_model.children())[:4])

        self.res_block1 = base_model.layer1 # 64 * 56 * 56
        self.res_block2 = base_model.layer2 # 128 * 28 * 28
        self.res_block3 = base_model.layer3 # 256 * 14 * 14
        self.res_block4 = base_model.layer4 # 512 * 7 * 7

        # Decoding Blocks
        if sum_mode:
            self.deconv_block1 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        else:
            self.deconv_block1 = nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(512, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Regressor
        # Dimensionality reduction layer
        self.fc_dim_reduce = nn.Sequential(
            nn.Linear(56 * 56 * 32, 2048),
            # nn.BatchNorm1d(2048)
        )
        # Translation vector prediction layer
        self.fc_position = nn.Sequential(
            nn.Linear(2048, 3),
            # nn.BatchNorm1d(3)
        )
        # Rotation quaternion prediction layer
        self.fc_rotation = nn.Sequential(
            nn.Linear(2048, self.ori_dim),
            # nn.BatchNorm1d(4)
        )

        # Initialize Weights
        init_modules = [self.deconv_block1, self.deconv_block2, self.deconv_block3, self.conv_block,
                        self.fc_dim_reduce, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, x):
        # Conv
        x = self.init_block(x)
        x_res1 = self.res_block1(x)
        x_res2 = self.res_block2(x_res1)
        x_res3 = self.res_block3(x_res2)
        x_res4 = self.res_block4(x_res3)

        # Deconv
        x_deconv1 = self.deconv_block1(x_res4)
        if self.sum_mode:
            x_deconv1 = x_res3 + x_deconv1
        else:
            x_deconv1 = torch.cat((x_res3, x_deconv1), dim=1)

        x_deconv2 = self.deconv_block2(x_deconv1)
        if self.sum_mode:
            x_deconv2 = x_res2 + x_deconv2
        else:
            x_deconv2 = torch.cat((x_res2, x_deconv2), dim=1)

        x_deconv3 = self.deconv_block3(x_deconv2)
        if self.sum_mode:
            x_deconv3 = x_res1 + x_deconv3
        else:
            x_deconv3 = torch.cat((x_res1, x_deconv3), dim=1)

        x_conv = self.conv_block(x_deconv3)
        x_linear = x_conv.view(x_conv.size(0), -1)
        x_fully = self.fc_dim_reduce(x_linear)
        x = F.relu(x_fully)

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training = dropout_on)

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully


class MobileNetV3Hourglass(nn.Module):
    def __init__(self, base_model, use_euler6, fixed_weight = False, sum_mode = True, dropout_rate = 0.0, bayesian = False):
        super(MobileNetV3Hourglass, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
        self.sum_mode = sum_mode
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # MobileNetV3 does not use `fc` but `classifier` for its final layers
        # self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.features = base_model.features

        if fixed_weight:
            for param in  nn.Sequential(*list(self.features.children())[:-1]).parameters():
                param.requires_grad = False

        # We manually determine these cut points by examining the network architecture
        # These indices represent the layers just before spatial reduction occurs
        # block_ends = [2, 4, 7, 13]  # Indices where feature size changes
        block_ends = [3, 6, 12, 16]

        # Splitting features into blocks
        self.block1 = nn.Sequential(*self.features[:block_ends[0] + 1]) # 24 * 56 * 56
        self.block2 = nn.Sequential(*self.features[block_ends[0] + 1:block_ends[1] + 1]) # 40 * 28 * 28
        self.block3 = nn.Sequential(*self.features[block_ends[1] + 1:block_ends[2] + 1]) # 112 * 14 * 14
        self.block4 = nn.Sequential(*self.features[block_ends[2] + 1:block_ends[3] + 1]) # 960 * 7 * 7

        # Decoding Blocks
        if sum_mode:
            self.deconv_block1 = nn.ConvTranspose2d(960, 112, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(112, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(40, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(24, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        else:
            self.deconv_block1 = nn.ConvTranspose2d(960, 112, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block2 = nn.ConvTranspose2d(224, 40, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.deconv_block3 = nn.ConvTranspose2d(80, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False, output_padding=1)
            self.conv_block = nn.Conv2d(48, 12, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

        # Regressor
        # Dimensionality reduction layer
        self.fc_dim_reduce = nn.Sequential(
            nn.Linear(56 * 56 * 12, 1280),
        )
        # Translation vector prediction layer
        self.fc_position = nn.Sequential(
            nn.Linear(1280, 3),
        )
        # Rotation quaternion prediction layer
        self.fc_rotation = nn.Sequential(
            nn.Linear(1280, self.ori_dim),
        )

        # Initialize Weights
        init_modules = [self.deconv_block1, self.deconv_block2, self.deconv_block3, self.conv_block,
                        self.fc_dim_reduce, self.fc_position, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x_1 = self.block1(x)
        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2)
        x_4 = self.block4(x_3)


        # Deconv
        x_deconv1 = self.deconv_block1(x_4)
        if self.sum_mode:
            x_deconv1 = x_3 + x_deconv1
        else:
            x_deconv1 = torch.cat((x_3, x_deconv1), dim=1)

        x_deconv2 = self.deconv_block2(x_deconv1)
        if self.sum_mode:
            x_deconv2 = x_2 + x_deconv2
        else:
            x_deconv2 = torch.cat((x_2, x_deconv2), dim=1)

        x_deconv3 = self.deconv_block3(x_deconv2)
        if self.sum_mode:
            x_deconv3 = x_1 + x_deconv3
        else:
            x_deconv3 = torch.cat((x_1, x_deconv3), dim=1)

        x_conv = self.conv_block(x_deconv3)
        x_linear = x_conv.view(x_conv.size(0), -1)
        x_fully = self.fc_dim_reduce(x_linear)
        x = F.relu(x_fully)

        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=dropout_on)

        position = self.fc_position(x)
        rotation = self.fc_rotation(x)

        return position, rotation, x_fully


class BranchResNet34(nn.Module):
    def __init__(self, base_model, use_euler6, fixed_weight=False, dropout_rate=0.0, bayesian = False):
        super(BranchResNet34, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate
        feat_in = base_model.fc.in_features

        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        # self.base_model = base_model

        if fixed_weight:
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.init_block = nn.Sequential(*list(base_model.children())[:4])

        # shared layers
        self.res_block1 = base_model.layer1 # 64 * 56 * 56
        self.res_block2 = base_model.layer2 # 128 * 28 * 28
        self.res_block3 = base_model.layer3 # 256 * 14 * 14

        # split layers for pos
        self.pos_res_block4 = copy.deepcopy(base_model.layer4) # 512 * 7 * 7
        self.pos_fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_position = nn.Linear(2048, 3, bias=True)

        # split layers for ori
        self.ori_res_block4 = copy.deepcopy(base_model.layer4)  # 512 * 7 * 7
        self.ori_fc_last = nn.Linear(feat_in, 2048, bias=True)
        self.fc_rotation = nn.Linear(2048, self.ori_dim, bias=True)

        init_modules = [self.pos_fc_last, self.fc_position, self.ori_fc_last, self.fc_rotation]

        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


    def forward(self, x):
        # shared layers
        x = self.init_block(x)
        x_res1 = self.res_block1(x)
        x_res2 = self.res_block2(x_res1)
        x_res3 = self.res_block3(x_res2)

        # split layers
        # pos
        x_pos_res4 = self.pos_res_block4(x_res3)
        x_pos = F.avg_pool2d(x_pos_res4, 7)
        x_pos = x_pos.view(x_pos.size(0), -1)
        x_pos_fully = self.pos_fc_last(x_pos)
        x_pos = F.relu(x_pos_fully)
        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x_pos = F.dropout(x_pos, p=self.dropout_rate, training=dropout_on)
        position = self.fc_position(x_pos)

        # ori
        x_ori_res4 = self.ori_res_block4(x_res3)
        x_ori = F.avg_pool2d(x_ori_res4, 7)
        x_ori = x_ori.view(x_ori.size(0), -1)
        x_ori_fully = self.pos_fc_last(x_ori)
        x_ori = F.relu(x_ori_fully)
        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x_ori = F.dropout(x_ori, p=self.dropout_rate, training=dropout_on)
        rotation = self.fc_rotation(x_ori)

        return position, rotation, torch.cat((x_pos_fully, x_ori_fully), dim=1)

class BranchMobileNetV3(nn.Module):
    def __init__(self, base_model, use_euler6, fixed_weight=False, dropout_rate=0.0, bayesian=False):
        super(BranchMobileNetV3, self).__init__()
        self.ori_dim = 6 if use_euler6 else 4
        self.bayesian = bayesian
        self.dropout_rate = dropout_rate

        # MobileNetV3 does not use `fc` but `classifier` for its final layers
        # self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        self.features = base_model.features
        feat_in = base_model.classifier[0].in_features # 960

        if fixed_weight:
            for param in nn.Sequential(*list(self.features.children())[:-1]).parameters():
                param.requires_grad = False

        # We manually determine these cut points by examining the network architecture
        # These indices represent the layers just before spatial reduction occurs
        block_ends = [3, 6, 12, 16]  # Indices where feature size changes

        # Splitting features into 4 blocks
        # shared blocks
        self.block1 = nn.Sequential(*self.features[:block_ends[0] + 1])  # 24 * 56 * 56
        self.block2 = nn.Sequential(*self.features[block_ends[0] + 1:block_ends[1] + 1])  # 40 * 28 * 28
        self.block3 = nn.Sequential(*self.features[block_ends[1] + 1:block_ends[2] + 1])  # 112 * 14 * 14

        self.block4 = nn.Sequential(*self.features[block_ends[2] + 1:block_ends[3] + 1])  # 960 * 7 * 7

        # split layers for pos
        self.pos_block4 = copy.deepcopy(self.block4)  # 960 * 7 * 7
        self.pos_fc_last = nn.Linear(feat_in, 1280, bias=True)
        self.fc_position = nn.Linear(1280, 3, bias=True)

        # split layers for ori
        self.ori_block4 = copy.deepcopy(self.block4)  # 512 * 7 * 7
        self.ori_fc_last = nn.Linear(feat_in, 1280, bias=True)
        self.fc_rotation = nn.Linear(1280, self.ori_dim, bias=True)



        # Initialize weights
        init_modules = [self.pos_fc_last, self.fc_position, self.ori_fc_last, self.fc_rotation]
        for module in init_modules:
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # shared layers
        x_1 = self.block1(x)
        x_2 = self.block2(x_1)
        x_3 = self.block3(x_2)

        # split layers
        # pos
        x_pos_4 = self.pos_block4(x_3)
        x_pos = F.avg_pool2d(x_pos_4, 7)
        x_pos = x_pos.view(x_pos.size(0), -1)
        x_pos_fully = self.pos_fc_last(x_pos)
        x_pos = F.relu(x_pos_fully)
        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x_pos = F.dropout(x_pos, p=self.dropout_rate, training=dropout_on)
        position = self.fc_position(x_pos)

        # ori
        x_ori_4 = self.ori_block4(x_3)
        x_ori = F.avg_pool2d(x_ori_4, 7)
        x_ori = x_ori.view(x_ori.size(0), -1)
        x_ori_fully = self.ori_fc_last(x_ori)
        x_ori = F.relu(x_ori_fully)
        dropout_on = self.training or self.bayesian
        if self.dropout_rate > 0:
            x_ori = F.dropout(x_ori, p=self.dropout_rate, training=dropout_on)
        rotation = self.fc_rotation(x_ori)

        return position, rotation, torch.cat((x_pos_fully, x_ori_fully), dim=1)
