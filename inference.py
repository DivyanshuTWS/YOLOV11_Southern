import os
from ultralytics import YOLO

# Load the trained YOLO classification model
model = YOLO(r"E:\Client POC\YOLOv11_POC\runs\classify\train4\weights\best.pt")

# Folder containing test images
image_folder = r"E:\Client POC\Yolo_Client_POC\dataset\val\SINGLE LINE"

for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder, filename)
        results = model(image_path, verbose=False)  # 👈 suppress internal logging

        probs = results[0].probs
        names = results[0].names

        if probs is not None:
            predicted_index = probs.data.argmax().item()
            predicted_class = names[predicted_index]
            print(f"{image_path} -> {predicted_class}")
