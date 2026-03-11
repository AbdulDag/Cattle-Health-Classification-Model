# 🐄 Cattle Health Classifier: Edge-AI Barn Monitoring

An intelligent, hybrid computer vision application designed to monitor cattle health in real-time. This system combines **YOLOv8x** for robust object tracking in crowded barn environments with a custom-trained **ResNet18** neural network to diagnose signs of lameness or illness based on posture and visual indicators.



## Key Features

* **Dual-Mode Analysis:** * **Snapshot Mode:** Drag-and-drop static image analysis for rapid, high-accuracy individual cow diagnostics.
  * **Live Video Feed:** Processes MP4 barn footage frame-by-frame, tracking multiple animals simultaneously.
* **Smart Bounding Box Filtering:** Automatically filters out "fractional" cows (e.g., heads poking through metal grates) using dynamic size and area algorithms to prevent false-positive classifications.
* **Color-Aware Diagnostics:** The custom ResNet18 model (V4) was specifically trained on full RGB imagery, allowing it to detect critical health indicators like coat condition and tongue/gum coloration.
* **Gradio Web Interface:** A clean, responsive UI that can be deployed locally or hosted on cloud platforms like Hugging Face Spaces.

## System Architecture

The pipeline utilizes a two-step "Crop and Classify" architecture to maximize accuracy on edge devices:

1. **Detection (YOLOv8x):** Scans the video frame, identifies all cattle, and draws precise bounding boxes.
2. **Preprocessing (PyTorch Transforms):** Extracts the bounding boxes. For video, it uses a non-destructive squish transform (`224x224`) to ensure vital anatomy (like a resting head or hooves) is not removed by center-cropping.
3. **Classification (ResNet18):** Analyzes the isolated crop and outputs a binary diagnostic probability (`Healthy` vs. `Sick`).

## 📊 Performance Metrics

The ResNet18 classification model was trained over 25 epochs, achieving high validation accuracy on a custom dataset of annotated cattle images. 


add here img

* **Detection Model:** YOLOv8x (Ultralytics)
* **Classification Model:** ResNet18 (PyTorch)
* **Confidence Threshold:** `0.35` for YOLO bounding boxes (optimized for crowd tracking without duplicate box generation).

## ⚙️ Local Setup & Installation

To run this application on your local machine (GPU recommended for video processing):

**1. Clone the repository:**
```bash
git clone [https://github.com/AbdulDag/Cattle-Health-Classification-Model.git](https://github.com/AbdulDag/Cattle-Health-Classification-Model.git)
cd Cattle-Health-Classification-Model

2. Install dependencies:
Make sure you have Python 3.8+ installed.

Bash
pip install -r requirements.txt
(Note: If you are using a CUDA-enabled NVIDIA GPU, install the appropriate PyTorch build from the official PyTorch website first).

3. Run the application:

Bash
python app.py
A local web server will start, and you can view the interface by navigating to http://127.0.0.1:7860 in your browser.

Note on Weights: The custom ResNet18 weights (best_cow_classifier_v4.pth) are included in the models/ directory. The YOLOv8x weights (yolov8x.pt) are omitted from version control due to file size limits, but will automatically download directly from Ultralytics the first time you run the script.

Usage
Snapshot Tab: Upload a clear, single image of a cow. The system will output a probability bar indicating health status.

Video Tab: Upload an .mp4 video. The system will process the footage and output a new .mp4 video with diagnostic bounding boxes tracking every detected animal.

```
🛠️ Built With
PyTorch & Torchvision
Ultralytics YOLO
Gradio
OpenCV
