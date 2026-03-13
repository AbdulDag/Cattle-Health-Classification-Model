import gradio as gr
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image
import os


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ['Healthy', 'Sick']

print("Loading YOLOv8x Mega Version for Video Tracking")
# Loading the biggest YOLO model for highly accurate tracking of multiple cows
yolo_model = YOLO('models/yolov8x.pt')

print("Loading ResNet18 V4")
# Load the base ResNet18 architecture but with NO pre-trained weights cuz I am loading my own
model = models.resnet18(weights=None)
# Get the number of inputs for the final Fully Connected (fc) layer
num_ftrs = model.fc.in_features

# Rebuilding the head of the model to fit my 2 classes instead of the default 1000
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(), # this function activates the neurons to learn non-linear patterns
    nn.Dropout(0.5), #Dropout regulariztion: randomly turns off 50% of neurons during training so the 
    # network is forced to actually learn features instead of just memorizing the dataset
    nn.Linear(128, len(class_names)) # final layer that outputs the probabilities for each class
)

#load custom v4 weights
model.load_state_dict(torch.load('models/best_cow_classifier_v4.pth', map_location=device, weights_only=True))
model = model.to(device)
model.eval() # Set model to evaluation mode (turns off dropout and batchnorm updates)


# Pipeline 1: For Snapshot Tab
val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),# Converts image to a PyTorch tensor (numbers between 0 and 1)
    # Normalize using standard ImageNet mean and std dev so the colors match what ResNet expects
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Pipeline 2: For Video YOLO Crops (squish)
video_crop_transforms = transforms.Compose([
    transforms.Resize((224, 224)), #use this instead of center crop so dont chop off any parts of the cow after yolo isolation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def predict_cow(image):
    #Image Tab: Full-Image Analysis
    if image is None:
        return None

    #pytorch was crashing because it expected a batch of images, so added with unsqueeze 0 a fake batch dimension    
    image_tensor = val_transforms(image).unsqueeze(0).to(device)

    #ignore gradients for testing to save memory
    with torch.no_grad():
        outputs = model(image_tensor)
        #get the probabilities for each class. sftmx fnction turns scores to percentages
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

     
    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

def analyze_video(video_path):
    #Video Tab: YOLO Cropping and Full-Video Analysis
    if video_path is None:
        return None, "Please upload a video."

  
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    #opencv video writer
    output_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cows_detected_total = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = yolo_model(frame, verbose=False, conf=0.35)
        cow_found_in_frame = False
        
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 19: # 19 is Cow YOLO class
                    cow_found_in_frame = True
                    cows_detected_total += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    cow_crop = frame[y1:y2, x1:x2]
                    if cow_crop.size > 0:
                        
                       #pytorch needs rgb to feed source to resnet
                        rgb_crop = cv2.cvtColor(cow_crop, cv2.COLOR_BGR2RGB)
                        pil_crop = Image.fromarray(rgb_crop)
                        
                        #sends the transformed image to the GPU
                        crop_tensor = video_crop_transforms(pil_crop).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = model(crop_tensor)
                            probs = torch.nn.functional.softmax(outputs[0], dim=0)
                            max_prob, predicted_idx = torch.max(probs, 0)
                            predicted_class = class_names[predicted_idx.item()]
                        
                        if "Sick" in predicted_class:
                            label = "Sick"
                            color = (0, 0, 255)   
                        else:
                            label = "Healthy"
                            color = (0, 255, 0)   
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        cv2.putText(frame, f"{label} ({max_prob.item()*100:.0f}%)", (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)

        if not cow_found_in_frame:
            cv2.putText(frame, "ANIMAL not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        out.write(frame)
        
    cap.release()
    out.release()
    
    if cows_detected_total == 0:
        return output_path, "ANIMAL not detected in entire video."
    else:
        return output_path, "Video processing complete."

#UI
custom_css = """
#app-container {
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
    padding-top: 2rem;
}
.center-text {
    text-align: center;
}
"""

#Gradio UI
with gr.Blocks(css=custom_css) as interface:
    with gr.Column(elem_id="app-container"):
        
        gr.Markdown("# Cattle Health Classifier", elem_classes="center-text")
        gr.Markdown("Upload a photo of a cow or live video feed to detect if it is showing signs of sickness. Powered by a custom ResNet18 Neural Network and YOLOv8.", elem_classes="center-text")
        
        with gr.Tabs(): #tabs at top
            with gr.TabItem("Snapshot Analysis"):
                with gr.Row():
                    image_input = gr.Image(type="pil", label="Upload Cow Photo")
                    label_output = gr.Label(num_top_classes=2, label="AI Diagnosis")
                
                predict_btn = gr.Button("Submit")
                #we link button to function 
                predict_btn.click(fn=predict_cow, inputs=image_input, outputs=label_output)

            with gr.TabItem("Live Video Feed"):
                with gr.Row():
                    vid_input = gr.Video(label="Upload Barn Footage")
                    vid_output = gr.Video(label="Analyzed Feed")
                vid_text = gr.Textbox(label="System Status")
                
                vid_btn = gr.Button("Analyze Video")
                vid_btn.click(fn=analyze_video, inputs=vid_input, outputs=[vid_output, vid_text])

if __name__ == "__main__":
    #launch the interface, if we set share=True it will generate a public link but it cannot handle mp4 input.
    interface.launch(share=True)