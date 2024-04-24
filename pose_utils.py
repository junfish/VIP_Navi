import torch
import numpy as np


def quat_to_euler(q, is_degree=False):
    w, x, y, z = q[0], q[1], q[2], q[3]

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(t3, t4)

    if is_degree:
        roll = np.rad2deg(roll)
        pitch = np.rad2deg(pitch)
        yaw = np.rad2deg(yaw)

    return np.array([roll, pitch, yaw])


def array_dist(pred, target):
    return np.linalg.norm(pred - target, 2)


def position_dist(pred, target):
    return np.linalg.norm(pred-target, 2)


def rotation_dist(pred, target):
    pred = quat_to_euler(pred)
    target = quat_to_euler(target)

    return np.linalg.norm(pred-target, 2)

def quat_dist(pred, target):
    # https://math.stackexchange.com/questions/3028462/rotation-distance
    return 2 * np.arccos(np.clip(np.abs(np.sum(pred * target)), 0, 1))

def fit_gaussian(pose_quat):
    # pose_quat = pose_quat.detach().cpu().numpy()

    num_data, _ = pose_quat.shape

    # Convert quat to euler
    # pose_euler = []
    # for i in range(0, num_data):
    #     pose = pose_quat[i, 4:]
    #     quat = pose_quat[i, :4]
    #     euler = quat_to_euler(quat)
    #     pose_euler.append(np.concatenate((euler, pose)))

    # Calculate mean and variance
    pose_quat_mean = np.mean(pose_quat, axis = 0)
    mat_var = np.zeros((7, 7))
    for i in range(0, num_data):
        pose_diff = pose_quat[i] - pose_quat_mean
        mat_var += pose_diff * np.transpose(pose_diff)

    mat_var = mat_var / num_data
    pose_quat_var = mat_var.diagonal()

    return pose_quat_mean, pose_quat_var








