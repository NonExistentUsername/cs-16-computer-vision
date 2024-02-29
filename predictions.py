from ultralytics import YOLO  # type: ignore

model = YOLO("yolo8nv3_trained.pt")


def predict(img):
    return model(img, device="mps", verbose=False)[0]
