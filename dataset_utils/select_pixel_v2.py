import cv2
import numpy as np
import glob
import os

# The code below is a simple software to annotate the image positions onto a predefined floor plan.
  # 1. "double click" to pinpoint a dot on map
  # 2. "press key 'A'" to collect the pixel-based coordinates.
  # 3. "prss key 'Esc'" to quit
# After three steps above, you will find the geo_coord.txt file in your colmap project directory.
# Open geo_coord.txt and add the time stamps for each image.

###########################
###### Change Floors ######
floor_name = "Basement" ####
###########################
###########################

folder_path = sorted(glob.glob("/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Localization/data/HST_video/Jun/train/" + floor_name + "/20*"))
current_project_path = folder_path[0:1] # select a colmap project
project_path = current_project_path[0]
dir_paths = glob.glob(os.path.join(project_path, 'DJI_20*')) + glob.glob(os.path.join(project_path, 'HAND_20*'))
dir_paths = [path for path in dir_paths if os.path.isdir(path)]
image_name = os.path.basename(dir_paths[0]) + '_frame_s.jpg'
# print(image_name)

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y), 7, (255,0,0),-1)
        mouseX,mouseY = x,y
# /Users/jasonyu/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LOWER-LEVEL_low.jpeg
# crop_BASEMENT_low.jpeg
# crop_LEVEL-ONE_low.jpeg
if floor_name == "Basement":
    img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_BASEMENT_low.jpeg')
elif floor_name == "Lower_Level":
    img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LOWER-LEVEL_low.jpeg')
elif floor_name == "Level_1":
    img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LEVEL-ONE_low.jpeg')
elif floor_name == "Level_2":
    img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LEVEL-TWO_low.jpeg')
# img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')

cv2.setMouseCallback('image', draw_circle)

with open(f'{project_path}/geo_coord.txt', 'w') as f:
    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('a'):
            # The below code will write image name, pixel x, pixel y, and 0 to the geo_coord.txt file. All we need to do is modify the time stamp of those coordinates in this file.
            f.write(f'{image_name} {mouseY} {mouseX} 0\n')
            print(image_name, mouseY, mouseX)