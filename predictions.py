from ultralytics import YOLO  # type: ignore

model = YOLO("yolov8n_trained.pt")


def predict(img):
    return model(img, device="mps")[0]
