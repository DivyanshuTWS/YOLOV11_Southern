This repository contains a proof-of-concept (POC) for image classification, prepared for a potential client Southern. The goal of this POC is to train a deep learning model capable of classifying engineering drawing images into their correct categories.

📌 Objective
To develop an image classification model that can accurately classify Southern's engineering drawing images into predefined categories using various YOLOv11 model variants.

🛠️ Model Variants Trained
Three different YOLOv11 variants were trained and evaluated for this task:

Model	Training Data	Epochs	Validation Accuracy	Notes
YOLOv11n	Augmented Dataset	30	79.92%	Lightweight model with decent performance
YOLOv11s	Augmented Dataset	30	~81%	Slightly better performance
YOLOv11m	Original Dataset Only	30	83%	Best performance overall

The best performing model was YOLOv11m, trained on the original dataset without augmentation.

📁 Model Weights
You can find the trained model weight files at the following locations:

YOLOv11n (best.pt): runs/classify/train2/weights/best.pt

YOLOv11s (best.pt): runs/classify/train3/weights/best.pt

YOLOv11m (best.pt): runs/classify/train4/weights/best.pt

🚀 Inference
To perform inference on a folder of class images, use the provided inference.py script.
This script will automatically classify each image and print predictions.
python inference.py 

📦 Requirements
Install the necessary dependencies with:
pip install ultralytics
