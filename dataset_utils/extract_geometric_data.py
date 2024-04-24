import os
import torch
import numpy as np
import json, pickle
from numpy.linalg import inv
import cv2
import tqdm
from PIL import Image
from colmap.scripts.python.read_write_model import read_model
from world_coordinates import quaternion_R_matrix
from kornia.geometry.conversions import rotation_matrix_to_quaternion, quaternion_to_rotation_matrix
import matplotlib.pyplot as plt

        # train_global_depths = torch.sort(torch.hstack(train_global_depths)).values
        # test_global_depths = torch.sort(torch.hstack(test_global_depths)).values
        # self.train_global_xmin = train_global_depths[int(xmin_percentile * (train_global_depths.shape[0] - 1))]
        # self.train_global_xmax = train_global_depths[int(xmax_percentile * (train_global_depths.shape[0] - 1))]
        # self.test_global_xmin = test_global_depths[int(xmin_percentile * (test_global_depths.shape[0] - 1))]
        # self.test_global_xmax = test_global_depths[int(xmax_percentile * (test_global_depths.shape[0] - 1))]
        # self.train_data = train_data
        # self.test_data = test_data

# Kendall, Alex, and Roberto Cipolla. "Geometric loss functions for camera pose regression with deep learning." In Proceedings of the IEEE CVPR, pp. 5974-5983. 2017.

floor_name = "Basement"
floor_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/" + floor_name


data = {}
for key in ['train']:
    image_train = os.path.join(floor_path, "image_" + key + "_all.txt")
    # xmin_p = 0.025
    # xmax_p = 0.975
    image_lines = open(image_train, 'r').readlines()[2:]
    colmap_proj_name = ""
    data_list, global_depths = [], []
    for image_line in tqdm.tqdm(image_lines):
        image_path, img_id = image_line.split(',')[0].strip(), int(image_line.split(',')[1].strip())
        colmap_model_path = os.path.join(floor_path, image_path.split('/')[0], "sparse", "geo")
        if colmap_proj_name != image_path.split('/')[0]:
            cameras, images, points3D = read_model(colmap_model_path)
            colmap_proj_name = image_path.split('/')[0]
        im = cv2.imread(os.path.join(floor_path, image_path))
        image = images[img_id]
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
            alpha=0,
            centerPrincipalPoint = True
        )
        new_K = torch.tensor(new_K)
        new_K[0, 2] = camera.width / 2
        new_K[1, 2] = camera.height / 2

        # im = cv2.undistort(im, K, dist_coeffs, newCameraMatrix=new_K.numpy()) # undistort image and center the principle point
        # im = preprocess(Image.fromarray(im[:, :, ::-1]))

        # w_t_c = torch.tensor(image.tvec).view(3, 1) # world coordinate system on camera coordinate
        # w_q_c = torch.tensor(image.qvec)
        #
        # R = torch.tensor(quaternion_R_matrix(w_q_c), dtype = torch.float64) # R in Eq.(6)
        # c_t_w = - R.T @ w_t_c # x in Eq.(6)
        #
        #
        # scene_coordinates = torch.zeros(max(points3D.keys()) + 1, 3, dtype = torch.float64)
        # for i, point3D in points3D.items():
        #     scene_coordinates[i] = torch.tensor(point3D.xyz)
        #
        # w_P = scene_coordinates[[i for i in image.point3D_ids if i != -1]] # G in Eq.(7); of size (|G|, 3)
        # c_p = new_K @ (R @ w_P.T + c_t_w)
        # c_p = c_p[:2] / c_p[2]
        # plt.scatter(c_p[0], c_p[1], color='red', s=100, marker='o')
        # plt.show()

        c_t_w = torch.tensor(image.tvec).view(3, 1) # world coordinate system on camera coordinate; x in Eq.(6)
        c_q_w = torch.tensor(image.qvec)

        if c_q_w[0] < 0:
            c_q_w *= -1

        R = torch.tensor(quaternion_R_matrix(c_q_w), dtype=torch.float64) # same to R; R in Eq.(6)
        c_R_w = quaternion_to_rotation_matrix(c_q_w)
        w_t_c = -c_R_w.T @ c_t_w

        scene_coordinates = torch.zeros(max(points3D.keys()) + 1, 3, dtype = torch.float64)
        for i, point3D in points3D.items():
            scene_coordinates[i] = torch.tensor(point3D.xyz)

        w_P = scene_coordinates[[i for i in image.point3D_ids if i != -1]]
        c_P = c_R_w @ (w_P.T - w_t_c)
        c_p = new_K @ c_P
        c_p = c_p[:2] / c_p[2]

        # plt.scatter(c_p[0], c_p[1], color='red', s=100, marker='o')
        # plt.show()
        depths = torch.sort(c_P[2]).values
        global_depths.append(depths.float())
        data_list.append({
            'image_path': image_path,
            'w_t_c': w_t_c.float(),
            'c_q_w': c_q_w.float(),
            'c_R_w': c_R_w.float(),
            'w_P': w_P.float(),
            'c_p': c_p.T.float(),
            'K': new_K.float(),
            # 'xmin': depths[int(xmin_p * (depths.shape[0] - 1))].float(),
            # 'xmax': depths[int(xmax_p * (depths.shape[0] - 1))].float()
        })
    data[key] = data_list
    # data[key + "_global_xmin"] = global_depths[int(xmin_p * (global_depths.shape[0] - 1))]
    # data[key + "_global_xmax"] = global_depths[int(xmax_p* (global_depths.shape[0] - 1))]

# Save using pickle
with open(os.path.join(floor_path, 'geometric_data.pkl'), 'wb') as f:
    pickle.dump(data, f)