import cv2
import numpy as np

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y), 7, (255,0,0),-1)
        mouseX,mouseY = x,y
# /Users/jasonyu/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LOWER-LEVEL_low.jpeg
# crop_BASEMENT_low.jpeg
# crop_LOWER-LEVEL_low.jpeg
# crop_LEVEL-ONE_low.jpeg
# crop_LEVEL-TWO_low.jpeg
img = cv2.imread('/data/juy220/LU Student Dropbox/Jun Yu/_Vinod/Indoor_Navi/Floor Plan/crop_LEVEL-TWO_low.jpeg')
# img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')

cv2.setMouseCallback('image', draw_circle)

while(1):
    cv2.imshow('image', img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print(mouseY, mouseX)
