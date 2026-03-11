import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cpu")
class_names = ['Healthy', 'Sick']

#Build and load model
resnet_model = models.resnet18(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, len(class_names))
)
# Ensure  .pth file is in the same folder
resnet_model.load_state_dict(torch.load('best_cow_classifier_v5.pth', map_location=device, weights_only=True))
resnet_model.eval()

#Define the transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#my function 
def get_diagnosis(cow_crop_bgr):
    #Takes an OpenCV image crop, runs it through ResNet18, and returns the diagnosis.
    # Translate BGR to RGB
    rgb_crop = cv2.cvtColor(cow_crop_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_crop)
    
    # Apply transforms and add the batch dimension we unsqueeze cuz it expects 32 images at once and we only sending 1 at a time. 
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # resnet diagnoses
    with torch.no_grad():
        outputs = resnet_model(input_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        conf, predicted_idx = torch.max(probs, 0)
        
    prediction = class_names[predicted_idx.item()]
    conf_pct = conf.item() * 100
    
    return prediction, conf_pct