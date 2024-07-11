import numpy as np
from numpy.linalg import inv
import os
import glob
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    # Basement
    # Lower_Level
    # Floor_3
    # test/Basement
    ###########################
    ###### Change Floors ######
    floor_name = "test/Level_1"  ###
    ###########################
    ###########################
    floor_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/" + floor_name
    f = open(os.path.join(floor_path, "image_test_all.txt"), 'w')
    f.write("Lehigh Health Science and Technology (HST) Building (https://www2.lehigh.edu/news/new-health-science-and-technology-building-a-hub-for-interdisciplinary-research).\n")
    # f.write("The Lower-Level Floor features a two-tiered structure.\n")
    f.write("IMG_PATH, IMG_ID, QW, QX, QY, QZ, TX, TY, TZ\n")
    folder_path = sorted(glob.glob(floor_path + "/20*"))
    total_image_idx = 0
    for proj_index, proj_path in enumerate(folder_path[0:25]):
        print(f"Working at {proj_index}-th project folder: {proj_path}...")
        file2read = open(os.path.join(proj_path, "camera2world_6DoF.txt"), 'r')
        lines = file2read.readlines()
        for idx, line in enumerate(lines):
            if idx > 0:
                IMG_ID, QW, QX, QY, QZ, TX, TY, TZ, NAME = line.split(',')
                img_id = int(IMG_ID)
                qw, qx, qy, qz, tx, ty, tz = float(QW), float(QX), float(QY), float(QZ), float(TX), float(TY), float(TZ)
                f.write('%-40s, %4.0f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f, %.8f\n' % (NAME.strip(), img_id, qw, qx, qy, qz, tx, ty, tz))
                total_image_idx += 1
    print("Total %s images in the training set." % total_image_idx)
    f.close()
