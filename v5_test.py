import cv2
import numpy as np
from ultralytics import YOLO as yolo
import sys
import torch 

s = 0
#load weights from .pth file
state_dict = torch.load('best_cow_classivier_v4')

#yolo model with architecture i wanna use
model = yolo('yolov8n.yaml')

#load .pth weights
model.model.load_state_dict(state_dict)

#save it as a.pt file for future use
model.save('converted_model.ptv4')

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





