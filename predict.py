import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

def predict_cow(image_path, model_path='best_cow_classifier_v4.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    # 1. Exact same image processing used in training
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. Rebuild the V4 Architecture
    class_names = ['0_Healthy', '1_Sick']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, len(class_names))
    )
    
    # 3. Load the trained brain
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure 'best_cow_classifier_v4.pth' is in the same folder.")
        return
        
    model = model.to(device)
    model.eval()

    # 4. Prep image and predict
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    prediction = class_names[predicted_idx.item()]
    confidence_pct = confidence.item() * 100
    
    print("\n" + "="*40)
    print(f"Analysing: {os.path.basename(image_path)}")
    print(f"Diagnosis: {prediction.replace('0_', '').replace('1_', '').upper()}")
    print(f"Confidence: {confidence_pct:.2f}%")
    print("="*40 + "\n")

if __name__ == '__main__':
    test_image = "test_cow9.jpg"
    predict_cow(test_image)