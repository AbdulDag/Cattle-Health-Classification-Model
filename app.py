import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. Load the Model Architecture
device = torch.device("cpu")
class_names = ['Healthy', 'Sick']

model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, len(class_names))
)

# Load your specific V4 brain
model.load_state_dict(torch.load('best_cow_classifier_v4.pth', map_location=device, weights_only=True))
model.eval()

# 2. Define the Prediction Function
def predict_cow(image):
    if image is None:
        return None
        
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
    return {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

# 3. Build the Clean, Centered Web UI
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

with gr.Blocks(css=custom_css) as interface:
    with gr.Column(elem_id="app-container"):
        
        # Original Headers, Centered
        gr.Markdown("# Cattle Health Classifier", elem_classes="center-text")
        gr.Markdown("Upload a photo of a cow to detect if it is showing signs of sickness. Powered by a custom ResNet18 Neural Network.", elem_classes="center-text")
        
        # Side-by-Side Original Inputs
        with gr.Row():
            image_input = gr.Image(type="pil", label="Upload Cow Photo")
            label_output = gr.Label(num_top_classes=2, label="AI Diagnosis")
        
        # Original Button
        predict_btn = gr.Button("Submit")
        
        predict_btn.click(fn=predict_cow, inputs=image_input, outputs=label_output)

if __name__ == "__main__":
    interface.launch(css=custom_css, share=True)