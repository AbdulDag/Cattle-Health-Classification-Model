import cv2
import numpy as np
from ultralytics import YOLO as yolo
import sys
import torch 

s = 0

#pipeline blueprint: 
"""
1. first process the video of the cow 
2. have yolo model apply bounding boxes on cows

 with its pre trained model on cow detection and spit out coords
3. use coords to crop photo and cut image into cow only pics
4. resnet18 model classifies each unique ID of cow and calculates ratio of
--healthy classification to sick and from there outputs a final value--
on each cow. 
5. openCV draws the box after and writes diagnosis on original image

"""
#changes source to sys.arv[1]
#e.g if python myscript.py cow_video.mp4 (cow_video.mp3 is sys.argv[1])
if len(sys.argv) > 1: 
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name ='video'
#initialize a window
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#download weights
cow_finder = yolo('yolov8n.pt')


while cv2.waitKey(1) != 27: #escape key
    has_frame, frame = source.read()
    #after grabbing a frame give to yolo which returns a list of result objects
    results = cow_finder(frame)

    #for loop: loop through bounding box , then check class id and if class id is 19 extract coords and print coords
    for result in results: 
        boxes = result.boxed #boxes object of bounding box outputs
        masks = result.masks #masks objects for segmentation masks outputs

    if not has_frame: 
        break
    cv2.imshow(win_name, frame)


source.release()
cv2.destroyWindow(win_name)





