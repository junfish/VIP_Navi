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
                if 33.9 <= time_stamp <= 34.4:
                    #11.1 <= time_stamp <= 11.55 or 72.2 <= time_stamp <= 72.65:
                    print("time stamp to save frame: ", time_stamp)
                    # cv2.ROTATE_180 (DJI horizaontal)
                    # cv2.ROTATE_90_CLOCKWISE (HAND vertical)
                    # cv2.ROTATE_90_COUNTERCLOCKWISE (DJI vertical)
                    # frame = cv2.rotate(frame, cv2.ROTATE_180)
                    # ubuntu
                    # DJI vertical: None
                    # HAND vertical: cv2.ROTATE_180
                    # DJI horizontal: None
                    # HAND horizontal: None
                    cv2.imwrite(os.path.join(dest_dir, dest_dir.split("/")[-1] + "_frame_%06.2fs.jpg") % time_stamp, frame)
                    num_img += 1
            else:
                print("time stamp to save frame: ", time_stamp)
                # cv2.ROTATE_180 (DJI horizontal)
                # cv2.ROTATE_90_CLOCKWISE (HAND vertical)
                # cv2.ROTATE_90_COUNTERCLOCKWISE (DJI vertical)
                # frame = cv2.rotate(frame, cv2.ROTATE_180)
                # ubuntu
                # DJI vertical: cv2.ROTATE_180
                # DJI horizontal: None
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

    # video_path = "/Users/jasonyu/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/NaVIP/image/20240307_220429_proj/DJI_20240307_220429.MP4"
    # dest_dir = "/Users/jasonyu/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/NaVIP/image/20240307_220429_proj/frames"
    # video_path = "./HST_video/Jun/Upload/20240326_221524_proj/HAND_20240326_221524.MOV"
    # dest_dir = "./HST_video/Jun/Upload/20240326_221524_proj/HAND_pad_20240326_221524"
    video_path = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/test/20240605_110520_proj/HAND_20240605_110520.MOV"
    # Basement/20240326_113942_proj/HAND_20240326_113942.MP4
    dest_dir = "/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/test/20240605_110520_proj/HAND_pad_20240605_110520"
    video2img(video_path, dest_dir, pad = True, time_intvl = 0.02)
    # 0.04: 2
    # 0.05: 2
    # 0.06: 3
    # 0.07: 4
    # 0.20: 11
    # 0.21: 12
    # 0.22: 13
    # 0.24: 14
    # 0.25: 14
    # 0.26: 15
    # 0.27: 16
    # 0.28: 16
    # video2img(video_path, dest_dir, pad = True, time_intvl = 0.05)