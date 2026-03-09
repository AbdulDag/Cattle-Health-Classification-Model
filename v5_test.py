import cv2
import numpy as np
from ultralytics import YOLO
import sys
s = 0

#changes source to sys.arv[1]
#e.g if python myscript.py cow_video.mp4 (cow_video.mp3 is sys.argv[1])
if len(sys.argv) > 1: 
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name ='Camera'
#initialize a window
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while cv2.waitKey(1) != 27: #escape key
    has_frame, frame = source.read()
    if not has_frame: 
        break
    cv2.imshow(win_name, frame)


source.release()
cv2.destroyWindow(win_name)





