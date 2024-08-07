###############
# Jun's version
###############

import cv2
import os
import numpy as np

def video2img(video_path, dest_dir, pad = False, time_intvl = 1):
    num_img = 0
    video_cap = cv2.VideoCapture(video_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    print("Video FPS (#frame/second): %.4f" % fps)
    print("user specified time interval: ", time_intvl)
    frame_intvl =  fps * time_intvl
    print("fps * time_intvl = %.4f" % frame_intvl)
    count = 0
    success = True
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    while success:
        success, frame = video_cap.read()
        count += 1
        time_stamp = count / fps
        # print("time stamp current frame:", time_stamp)
        if (count + 0) % int(frame_intvl) == 0 and success:
            if pad:
                if 103.5 <= time_stamp <= 103.9:
                    print("time stamp to save frame: ", time_stamp)
                    # cv2.ROTATE_180 (DJI horizontal)
                    # cv2.ROTATE_90_CLOCKWISE (HAND vertical)
                    # cv2.ROTATE_90_COUNTERCLOCKWISE (DJI vertical)
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    cv2.imwrite(os.path.join(dest_dir, dest_dir.split("/")[-1] + "_frame_%06.2fs.jpg") % time_stamp, frame)
                    num_img += 1
            else:
                print("time stamp to save frame: ", time_stamp)
                # cv2.ROTATE_180 (DJI horizaontal)
                # cv2.ROTATE_90_CLOCKWISE (HAND vertical)
                # cv2.ROTATE_90_COUNTERCLOCKWISE (DJI vertical)
                # frame = cv2.rotate(frame, cv2.ROTATE_180)
                # ubuntu
                # DJI vertical: None
                # HAND vertical: None
                # DJI horizontal: None
                # HAND horizontal: None
                cv2.imwrite(os.path.join(dest_dir, dest_dir.split("/")[-1] + "_frame_%05.1fs.jpg") % time_stamp, frame)
                num_img += 1
    video_cap.release()
    print("Total %4.0f images." % num_img)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # /HST_video/Jun/Basement
    # /HST_video/2160_60fps
    # /HST_video/Jun/Floor_2
    # /HST_video/Jun/Floor_3
    # /HST_video/Yifan/Floor_2
    # /HST_video/Yifan/Floor_3
    floor_name = 'test'
    # file_name_list= ['20240301_002758', '20240301_003050', '20240301_003422', '20240301_003629', '20240303_000805', '20240303_001013', '20240303_001227',
    #                  '20240303_001602', '20240305_101535', '20240305_101757', '20240314_235122', '20240314_235334', '20240316_220837', '20240316_221120',
    #                  '20240316_221350', '20240326_213822', '20240326_214038', '20240326_214245', '20240222_121220']
    file_name_list = ['20240604_181202', '20240604_181610', '20240604_182151', '20240604_182609', '20240605_010606', '20240605_010952', '20240605_105220', '20240605_105632', '20240605_110144', '20240605_110520', '20240605_111006', '20240605_111413']
    for file_name in file_name_list:
        video_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/" + floor_name + "/" + file_name + "_proj/HAND_" + file_name + ".MOV"
        dest_dir = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/" + floor_name + '/' + file_name + "_proj/HAND_" + file_name
        video2img(video_path, dest_dir, pad = False, time_intvl = 0.26)
        # 0.05: 2
        # 0.20: 11
        # 0.21: 12
        # 0.22: 13
        # 0.23: 13
        # 0.24: 14
        # 0.25: 14
        # 0.26: 15
        # 0.27: 16
        # 0.28: 16
        # 0.29: 17
    # video2img(video_path, dest_dir, pad = True, time_intvl = 0.05)