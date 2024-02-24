from ultralytics import YOLO  # type: ignore

# Load a model
model = YOLO("yolov8n_trained.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(
    data="./datasets/data.yaml",
    epochs=24,
    imgsz=640,
    device="mps",
)

try:
    # Evaluate the model
    results = model.evaluate()  # results are stored in 'results.txt'
except Exception as e:
    print(f"An error occurred while evaluating the model: {e}")


# Save the model
model.save("yolov8n_trained_v2.pt")
