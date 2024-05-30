from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("runs/penny/weights/best.pt")

# Run inference on 'bus.jpg' with arguments
model.predict("inference/penny3.jpg", save=True, imgsz=320, conf=0.5)