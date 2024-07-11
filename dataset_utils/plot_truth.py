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

def R_matrix2vector(rot_matrix):
    # Convert quaternion to rotation matrix
    # rotation_matrix = R.from_quat(Q).as_matrix()

    # Convert rotation matrix to rotation vector
    rot_vector, _ = cv2.Rodrigues(rot_matrix)
    return rot_vector

def plot3d(x, y, z, dest_dir):
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # ax.stem(x, y, z)
    ax.scatter3D(x, y, z, s=2, marker='o', color="red")
    plt.show()
    # plt.savefig(os.path.join(dest_dir, 'path_on_anchor.png'))

def plot2D_plane(a, b, c, d):

    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)

    X, Y = np.meshgrid(x, y)
    Z = (d - a * X - b * Y) / c

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, Z)

if __name__ == "__main__":
    ###########################
    ###### Change Floors ######
    floor_name = "Level_1"  ###
    ###########################
    ###########################
    if floor_name == "Basement":
        img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_BASEMENT_low.jpeg')
    elif floor_name == "Lower_Level":
        img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LOWER-LEVEL_low.jpeg')
    elif floor_name == "Level_1":
        img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LEVEL-ONE_low.jpeg')
    elif floor_name == "Level_2":
        img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LEVEL-TWO_low.jpeg')
    r = 4
    ### BGR
    color = (215, 155, 66)
    # (133, 173, 235) # orange
    # (149, 228, 177) # green
    # (255, 133, 233) # purple
    # (177, 235, 133) # blue
    folder_path = sorted(glob.glob("/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/test/" + floor_name + "/20*"))
    alpha = 0.6 # Transparency factor.
    all_image_num = 0
    for proj_path in folder_path[24:25]:
        print(f"Working at {proj_path}...")
        overlay = img.copy()
        cam2world_lines = open(os.path.join(proj_path, "camera2world_6DoF.txt"), 'r').readlines()
        print("%3.0f images are annotated in this project." % (len(cam2world_lines) - 1))
        all_image_num += (len(cam2world_lines) - 1)
        x_list, y_list, z_list = [], [], []
        for image_index, image_line in enumerate(cam2world_lines):
            if image_index >= 1:
                image_index, iqw, iqx, iqy, iqz, itx, ity, itz, image_name = image_line.split(', ')
                iqw, iqx, iqy, iqz, itx, ity, itz = float(iqw), float(iqx), float(iqy), float(iqz), float(itx), float(ity), float(itz)
                cv2.circle(overlay, (int(ity), int(itx)), r, color, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


    cv2.imshow("some", img)
    cv2.waitKey(0)
    print("%3.0f total images are annotated." % all_image_num)
    cv2.destroyAllWindows()
