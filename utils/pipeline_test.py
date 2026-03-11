import cv2
import numpy as np
from ultralytics import YOLO as yolo
import sys
import torch 
from PIL import Image

from cow_model import get_diagnosis


s = 0
found_cow = 0
#pipeline blueprint: 
"""
1. first process the video of the cow 
2. have yolo model apply bounding boxes on cows

 with its pre trained model on cow detection and spit out coords

 Path 1: Frontend: Bounding boxes drawn
 Path 2: backend: coords crop the photos into cow only pics which are then sent through pipeline to resnet18 model for analysis
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

win_name ='frontend'
#initialize a window
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
#download weights
cow_finder = yolo('yolov8n.pt')


while cv2.waitKey(1) != 27: #escape key
    has_frame, frame = source.read()
    #after grabbing a frame give to yolo which returns a list of result objects
    #moved break up so if frame is empty it leaves before doing the math
    if not has_frame: 
        break
    results = cow_finder(frame)

    #for loop: loop through bounding box , then check class id and if class id is 19 extract coords and print coords
    for result in results: 
        boxes = result.boxes #boxes object of bounding box outputs
        masks = result.masks #masks objects for segmentation masks outputs
        keypoints = result.keypoints
        probs = result.probs #probs object for classification output. we will use this
        obb = result.obb #oriented boxes object for OBB outputs

    
        for box in boxes: 
        #extract class ID
            class_id = int(box.cls[0])

            if class_id == 19: 
                #extract x1,x2,x2,y2 cords
                coords = box.xyxy[0]
                print(f"Cow Located at coordinates: {coords}")
                #cvrt to int cuz the tensor values are floats, numpy needs integers, it cant cut a fraction of a pixel.
                x1 = int(coords[0])
                y1 = int(coords[1])
                x2 = int(coords[2])
                y2 = int(coords[3])
                #crop based on matrix values
                #We pass slice instead of index like this: [start:end] credits to w3 schools
                cow_crop = frame[y1:y2, x1:x2]
                
                
                if cow_crop.size > 0:
                    cv2.imshow("Backend: Cropped Cow", cow_crop)
                    prediction, conf_pct = get_diagnosis(cow_crop)
                    box_color = (0, 255, 0) if prediction == 'Healthy' else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)
                    label = f"{prediction} ({conf_pct:.1f}%)"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            else: 
                #use later for frontend idk
                found_cow = 0
#figure out how to slice a 2D NumPy array in Python 
#this is because yolo store in matrix format of rows and coloumns where row = y and coloumns = x (height and weidth)
            
        cv2.imshow(win_name, frame)


source.release()
cv2.destroyWindow(win_name)





