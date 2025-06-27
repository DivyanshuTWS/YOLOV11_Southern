from ultralytics import YOLO

# Load a COCO-pretrained YOLO11s-cls model
model = YOLO("YOLO11m-cls.pt")

# Train the model with early stopping
results = model.train(
    data=r"E:\Client POC\Yolo_Client_POC\dataset",
    epochs=30,             # Total max epochs
    imgsz=640
)
