import numpy as np
import torch
from torchvision import transforms, models, datasets
import torch.nn as nn
from torchvision.models import (Inception_V3_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, MobileNet_V3_Large_Weights)
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from kornia.geometry.conversions import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix

def print_feature_size(my_model, dummy_input):
    features = my_model.features
    for i, feature in enumerate(features):
        output = feature(dummy_input)
        print(f"Layer {i}: feature size from {dummy_input.shape} to {output.shape}")
        dummy_input = output  # Feed the output back as the next input

def analyze_block_transitions(my_model, dummy_input):
    features = my_model.features
    current_size = dummy_input.shape[-1]
    for i, feature in enumerate(features):
        output = feature(dummy_input)
        new_size = output.shape[-1]
        if new_size < current_size:
            print(f"Block {output.shape} transition at layer {i}, size reduced from {dummy_input.shape} to {output.shape}")
            current_size = new_size
        dummy_input = output  # Feed the output back as the next input

dummy_input = torch.rand(1, 3, 224, 224)

# weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1
my_model_1 = models.mobilenet_v3_large()

print_feature_size(my_model_1, dummy_input)
analyze_block_transitions(my_model_1, dummy_input)

my_resnet_model = models.resnet34()
init_block = nn.Sequential(*list(my_resnet_model.children())[:4])
res_block1 = my_resnet_model.layer1
res_block2 = my_resnet_model.layer2
res_block3 = my_resnet_model.layer3
res_block4 = my_resnet_model.layer4

x = init_block(dummy_input)

x = res_block1(x)
print(x.shape)
x = res_block2(x)
print(x.shape)
x = res_block3(x)
print(x.shape)
x = res_block4(x)
print(x.shape)