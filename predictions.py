from ultralytics import YOLO  # type: ignore

model = YOLO("yolov8n.pt")


def predict(img):
    return model(img, device="mps")[0]
