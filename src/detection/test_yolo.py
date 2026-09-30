from ultralytics import YOLO
import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Using:", torch.cuda.get_device_name(0))
    device = 0
else:
    print("GPU unavailable. Using CPU.")
    device = "cpu"

model = YOLO("yolo11n.pt")

results = model(
    "https://ultralytics.com/images/bus.jpg",
    device=device
)

print("YOLO inference completed successfully!")