from ultralytics import YOLO  # type: ignore

model = YOLO("best.pt")


def predict(img):
    return model(img, device="mps", verbose=False)[0]
