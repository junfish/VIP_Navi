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
    # crop_LOWER-LEVEL_low.jpeg

    purple_bgr = (255, 133, 233)  # lower
    green_bgr = (149, 228, 177)  # basement
    orange_bgr = (163, 203, 235)  # 2
    blue_bgr = (215, 155, 66)  # 1
    red_bgr = (128, 128, 240) # pred

    img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LEVEL-TWO_low.jpeg')

    test_image_path = []
    test_image_lines = open("/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/test/Level_2/image_test_20.txt", 'r').readlines()
    for test_image_line_idx, test_image_line in enumerate(test_image_lines):
        if test_image_line_idx >= 2:
            test_image_path.append(test_image_line.split(',')[0].strip())

    ground_truth = open("/home/juy220/PycharmProjects/VIP_Navi/test_Resnet34_Level_2/2024-05-10-13:45:25/true.csv", 'r').readlines()
    predictions = open("/home/juy220/PycharmProjects/VIP_Navi/test_Resnet34_Level_2/2024-05-10-13:45:25/estim.csv", 'r').readlines()

    r = 3
    truth_color = (255, 133, 233)
    pred_color = (155, 233, 133)
    folder_path = sorted(glob.glob("/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/test/Level_2/20*"))
    alpha = 1 # Transparency factor.
    for proj_path in folder_path[18:19]:
        overlay = img.copy()
        cam2world_lines = open(os.path.join(proj_path, "camera2world_6DoF.txt"), 'r').readlines()
        print("%3.0f images are annotated in this project." % (len(cam2world_lines) - 1))
        x_list, y_list, z_list = [], [], []
        for image_index, image_line in enumerate(cam2world_lines):
            if image_index >= 1:
                _, _, _, _, _, _, _, _, image_name = image_line.split(', ')
                test_image_index = test_image_path.index(image_name.strip())
                # iqw, iqx, iqy, iqz, itx, ity, itz = float(iqw), float(iqx), float(iqy), float(iqz), float(itx), float(ity), float(itz)
                x_truth, y_truth, z_truth = ground_truth[test_image_index].split(',')[:3]
                x_pred, y_pred, z_pred = predictions[test_image_index].split(',')[:3]
                cv2.circle(overlay, (int(float(y_truth)), int(float(x_truth))), r, orange_bgr, -1)
                cv2.circle(overlay, (int(float(y_pred)), int(float(x_pred))), r, red_bgr, -1)
        img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)



    cv2.imshow("some", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
