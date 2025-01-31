# Leaf Detection Using YOLOv5

This repository contains code and resources for detecting leaves using the YOLOv5 object detection model. The project leverages deep learning techniques to accurately identify and classify leaves from images.

## Features
- Uses YOLOv5 for real-time leaf detection
- Supports training on custom datasets
- Compatible with GPU acceleration for faster inference
- Preprocessing and annotation support using Roboflow

## Usage
1. Setup and Configuration

Ensure that your environment is set up correctly:

import torch
import os
from IPython.display import Image, clear_output  

print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

2. Training the Model

Prepare your dataset and train YOLOv5 using the following command:

!python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov5s.pt

3. Running Inference

To detect leaves in images, use:

!python detect.py --source images/ --weights runs/train/exp/weights/best.pt --conf 0.4

Dataset

The dataset is prepared and annotated using Roboflow. Make sure to download and preprocess it before training.

Results

After training, the model can be used for real-time detection of leaves. Evaluation metrics such as mAP (mean Average Precision) can be used to assess model performance.

Acknowledgments

This project is based on the YOLOv5 framework developed by Ultralytics.

License

This project is open-source and available under the MIT License.

## Installation
Clone the repository and navigate to the project folder:
Usage
```bash
!git clone https://github.com/ultralytics/yolov5  
%cd yolov5
## Install the necessary dependencies:
%pip install -qr requirements.txt
%pip install -q roboflow

