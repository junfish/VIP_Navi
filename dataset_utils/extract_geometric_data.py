import os
import torch
import numpy as np
import json, pickle
from numpy.linalg import inv
import cv2
import tqdm
import glob
from PIL import Image
from colmap.scripts.python.read_write_model import read_model
from world_coordinates import quaternion_R_matrix
from kornia.geometry.conversions import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
# Kendall, Alex, and Roberto Cipolla. "Geometric loss functions for camera pose regression with deep learning." In Proceedings of the IEEE CVPR, pp. 5974-5983. 2017.

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

floor_name = "Lower_Level"
train_test_split = 90
floor_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/" + floor_name
colmap_folder_path = sorted(glob.glob(floor_path + "/20*"))

data = {}

# xmin_p = 0.025
# xmax_p = 0.975

data_list, global_depths = [], []

for proj_idx, colmap_proj_name in tqdm.tqdm(enumerate(colmap_folder_path)):
    colmap_model_path = os.path.join(colmap_proj_name, "sparse", "geo")
    image_folder_list = [d for d in os.listdir(colmap_proj_name) if os.path.isdir(os.path.join(colmap_proj_name, d)) and d.startswith(('DJI_20', 'HAND_20'))]
    image_folder = image_folder_list[0]
    cameras, images, points3D = read_model(colmap_model_path)
    scene_coordinates = torch.zeros(max(points3D.keys()) + 1, 3, dtype=torch.float64)
    for i, point3D in points3D.items():
        scene_coordinates[i] = torch.tensor(point3D.xyz)
    for img_id, image in images.items():
        im = cv2.imread(os.path.join(colmap_proj_name, image_folder, image.name))
        # cv2.imshow('image_1', im)
        # cv2.waitKey(0)
        camera = cameras[image.camera_id]
        f, cx, cy, k = camera.params
        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ])
        dist_coeffs = np.array([k, 0, 0, 0, 0])
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix = K,
            distCoeffs = dist_coeffs,
            imageSize = im.shape[:2][::-1],
            alpha = 0,
            centerPrincipalPoint = True
        )
        new_K = torch.tensor(new_K)
        new_K[0, 2] = camera.width / 2
        new_K[1, 2] = camera.height / 2

        # im = cv2.undistort(im, K, dist_coeffs, newCameraMatrix=new_K.numpy())
        # im = transform(Image.fromarray(im[:, :, ::-1]))

        c_t_w = torch.tensor(image.tvec).view(3, 1) # world coordinate system on camera coordinate; x in Eq.(6)
        c_q_w = torch.tensor(image.qvec)

        if c_q_w[0] < 0:
            c_q_w *= -1

        c_R_w = quaternion_to_rotation_matrix(c_q_w) # same to R; R in Eq.(6)
        # R = torch.tensor(quaternion_R_matrix(c_q_w), dtype=torch.float64)
        w_t_c = -c_R_w.T @ c_t_w


        w_P = scene_coordinates[[i for i in image.point3D_ids if i != -1]] # G in Eq.(7); of size (|G|, 3)
        print(w_P.shape[0])
        c_P = c_R_w @ (w_P.T - w_t_c) # (R @ g + x) in Eq.(6)
        c_p = new_K @ c_P # Eq.(6) --> K @ (R @ g + x)
        c_p = c_p[:2] / c_p[2]

        # plt.scatter(c_p[0], c_p[1], color='red', s=100, marker='o')
        # plt.show()
        # depths = torch.sort(c_P[2]).values
        # global_depths.append(depths.float())
        data_list.append({
            'image_path': os.path.join(colmap_proj_name.split('/')[-1], image_folder, image.name),
            'w_t_c': w_t_c.float(), # (3, 1)
            'c_q_w': c_q_w.float(),
            'c_R_w': c_R_w.float(),
            'w_P': w_P.float(),
            'c_p': c_p.T.float(),
            'K': new_K.float(),
            # 'xmin': depths[int(xmin_p * (depths.shape[0] - 1))].float(),
            # 'xmax': depths[int(xmax_p * (depths.shape[0] - 1))].float()
        })
    if proj_idx == (train_test_split - 1):
        data["train"] = data_list
        data_list, global_depths = [], []
data["test"] = data_list


    # data[key + "_global_xmin"] = global_depths[int(xmin_p * (global_depths.shape[0] - 1))]
    # data[key + "_global_xmax"] = global_depths[int(xmax_p* (global_depths.shape[0] - 1))]

# Save using pickle
with open(os.path.join(floor_path, 'geometric_data.pkl'), 'wb') as f:
    pickle.dump(data, f)