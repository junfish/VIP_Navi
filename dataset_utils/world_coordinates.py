import numpy as np
from numpy.linalg import inv
import os
import glob
import matplotlib.pyplot as plt
import cv2


def quaternion_R_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # R(Q) = [ 2(q0^2 + q1^2) - 1,   2(q1q2 - q0q3),       2(q1q3 + q0q2);
    #          2(q1q2 + q0q3),       2(q0^2 + q2^2) - 1,   2(q2q3 - q0q1);
    #          2(q1q3 - q0q2),       2(q2q3 + q0q1),       2(q0^2 + q3^2) - 1
    #        ]


    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def plot3d(x_list, y_list, z_list, q_list, dest_dir):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    for x, y, z, q in zip(x_list, y_list, z_list, q_list):
        direction_vector = quaternion_R_matrix(q)[:, 2]  # Using the z-axis as direction

        # Plot the point
        ax.scatter(x, y, z, s = 2, marker='o', color = "red")
        # Plot the arrow for direction
        # ax.quiver(x, y, z, direction_vector[0], direction_vector[1], direction_vector[2], length = 2, color = 'blue')
    # ax.scatter3D(x, y, z, s = 2, marker='o', color = "red")
    # ax.quiver(point[0], point[1], point[2], direction_vector[0], direction_vector[1], direction_vector[2], length=0.5, color='red')
    # ax.scatter(0, 0, 0, s=55, marker='o', color="green")
    # ax.quiver(0, 0, 0, quaternion_R_matrix([1, 0, 0, 0])[:, 2][0], quaternion_R_matrix([1, 0, 0, 0])[:, 2][1], quaternion_R_matrix([1, 0, 0, 0])[:, 2][2], length=6, color='green')
    ax.set_xlim([0, 1200])
    ax.set_ylim([0, 2600])
    ax.set_zlim([-40, 100])
    # plt.show()
    plt.savefig(os.path.join(dest_dir, 'path_stem.png'))
    plt.close(fig)


if __name__ == "__main__":

    ###########################
    ###### Change Floors ######
    floor_name = "Level_1"  ###
    ###########################
    ###########################
    # read and sort all colmap projects (e.g., ./Basement/20231220_141254_proj)
    folder_path = sorted(glob.glob("/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/test/" + floor_name + "/20*"))

    for proj_index, proj_path in enumerate(folder_path[24:25]): # for-loop all colmap projects

        print(f"Working at {proj_index}-th project folder: {proj_path}...")
        proj_name = proj_path.split('/')[-1] # obtain the folder name, e.g., 20231220_141254_proj
        file2read = open(os.path.join(proj_path, "images.txt"), 'r') # read the image.txt (easily exported as txt by colmap)
        # For tidy publishing purposes, the file image.txt has been relocated to the directory 20231220_141254_proj/sparse/geo/images.txt. Please remember to change the file path to your own directory.

        lines = file2read.readlines() # read lines stored in image.txt
        f1 = open(os.path.join(proj_path, "camera2world_6DoF.txt"), 'w') # file to store clean version of camera pose (qw, qx, qy, qz, tx, ty, tz)

        x_list, y_list, z_list, q_list = [], [], [], []
        for idx, line in enumerate(lines):
            if idx == 1:
                f1.write("IMG_ID, QW, QX, QY, QZ, TX, TY, TZ, IMG_PATH\n")
            if idx > 3 and idx % 2 == 0 and len(line.split(' ')) == 10:
                Quaternion = []
                IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME = line.split(' ')
                image_folder_name = "_".join(NAME.split('_')[:3])
                img_id = int(IMAGE_ID)
                qw, qx, qy, qz, tx, ty, tz = float(QW), float(QX), float(QY), float(QZ), float(TX), float(TY), float(TZ)
                if qw < 0:
                    qw, qx, qy, qz = -qw, -qx, -qy, -qz
                cam_id = int(CAMERA_ID)
                # calculate the actual coordinates
                R_matrix = quaternion_R_matrix([qw, qx, qy, qz])
                x, y, z = -np.matmul(inv(R_matrix), np.array([[tx], [ty], [tz]])).squeeze()

                x_list.append(x)
                y_list.append(y)
                z_list.append(z)
                q_list.append([qw, qx, qy, qz])
                if "anchor" in NAME:
                    pass
                else:
                    f1.write('%3.0f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %-40s\n' % (img_id, qw, -qx, -qy, -qz, x, y, z, '/'.join([proj_name, image_folder_name, NAME.split('/')[-1].strip()])))

        f1.close()
        plot3d(x_list, y_list, z_list, q_list, proj_path) # visualize the camera pose in a 3D space


