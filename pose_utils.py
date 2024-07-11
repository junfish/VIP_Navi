import torch
import math
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


def quaternion_to_euler6(q):
    """
    Convert a batch of quaternions into euler angles (roll, pitch, yaw) represented as 6 sin and cos values.
    The input q is expected to have the shape (batch_size, 4) where each row is a quaternion [w, x, y, z].
    The output is an array of shape (batch_size, 6) containing [sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw] for each quaternion.
    """
    # Split the quaternion into its components
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Calculate the sin and cos of the roll angles
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = torch.atan2(t0, t1)
    sin_roll = torch.sin(roll_x)
    cos_roll = torch.cos(roll_x)

    # Calculate the sin and cos of the pitch angles
    t2 = 2.0 * (w * y - z * x)
    t2 = torch.clamp(t2, -1.0, 1.0)  # Clamp to avoid errors at the poles
    pitch_y = torch.asin(t2)
    sin_pitch = torch.sin(pitch_y)
    cos_pitch = torch.cos(pitch_y)

    # Calculate the sin and cos of the yaw angles
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = torch.atan2(t3, t4)
    sin_yaw = torch.sin(yaw_z)
    cos_yaw = torch.cos(yaw_z)

    # Combine the results and scale by sqrt(2) as needed
    result = torch.stack((sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw), dim = -1)

    return torch.sqrt(torch.tensor(2.0)) * result

def euler6_to_quaternion(e): # for testing process (batch_size = 1), no batch size
    """
    Convert Euler angles in euler6 format to a quaternion.
    """

    # Reconstruct Euler angles from sin and cos values
    sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw = e[0], e[1], e[2], e[3], e[4], e[5]
    roll = np.arctan2(sin_roll, cos_roll)
    pitch = np.arctan2(sin_pitch, cos_pitch)
    yaw = np.arctan2(sin_yaw, cos_yaw)

    # Compute half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Compute quaternion components
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return np.array([w, x, y, z])


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

def euler_distance(pred, target):
    """
    Calculate the rotation distance between two directions given in euler6 components.
    dir1 and dir2 are tuples in the form (sin_roll, cos_roll, sin_pitch, cos_pitch, sin_yaw, cos_yaw).
    """
    # Reconstruct angles
    roll_pred, pitch_pred, yaw_pred = math.atan2(pred[0], pred[1]), math.atan2(pred[2], pred[3]), math.atan2(pred[4], pred[5])
    roll_target, pitch_target, yaw_target = math.atan2(target[0], target[1]), math.atan2(target[2], target[3]), math.atan2(target[4], target[5])

    # Calculate differences
    roll_diff = abs(roll_pred - roll_target)
    pitch_diff = abs(pitch_pred - pitch_target)
    yaw_diff = abs(yaw_pred - yaw_target)

    # Normalize the differences
    roll_diff = roll_diff if roll_diff <= math.pi else 2 * math.pi - roll_diff
    pitch_diff = pitch_diff if pitch_diff <= math.pi else 2 * math.pi - pitch_diff
    yaw_diff = yaw_diff if yaw_diff <= math.pi else 2 * math.pi - yaw_diff

    # Sum of differences as a simple rotational distance measure
    return roll_diff + pitch_diff + yaw_diff

def fit_gaussian(pose_quat):
    # pose_quat = pose_quat.detach().cpu().numpy()

    num_data, _ = pose_quat.shape


    # Calculate mean and variance
    pose_quat_mean = np.mean(pose_quat, axis = 0)
    mat_var = np.zeros((9, 9))
    for i in range(0, num_data):
        pose_diff = pose_quat[i] - pose_quat_mean
        mat_var += pose_diff * np.transpose(pose_diff)

    mat_var = mat_var / num_data
    pose_quat_var = mat_var.diagonal()

    return pose_quat_mean, pose_quat_var








