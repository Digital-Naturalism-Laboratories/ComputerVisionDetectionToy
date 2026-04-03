from ultralytics import YOLO

# Load your custom model
model = YOLO("yolo11m_4500_imgsz1600_b1_2024-01-18.pt")

# Export to ONNX with static shapes
# imgsz should match the size you used during training (e.g., 640)
path = model.export(format="onnx", imgsz=640, dynamic=False, opset=12)

print(f"Model exported to: {path}")