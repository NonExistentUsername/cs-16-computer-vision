import json
import os

import cv2
import numpy as np

from predictions import predict

categories = [
    {"supercategory": "ct", "id": 1, "name": "ct_body"},
    {"supercategory": "ct", "id": 2, "name": "ct_head"},
    {"supercategory": "t", "id": 3, "name": "t_body"},
    {"supercategory": "t", "id": 4, "name": "t_head"},
]

images = []

# Default values
images_license = 0
height = 640
width = 640
date_captured = None

annotations = []  # type: ignore

# Default values
segmentation: list = []
iscrowd = 0
attributes = {"occluded": False, "rotation": 0.0}


def calculate_area(bbox: list) -> float:
    return bbox[2] * bbox[3]


def add_image(file_name: str, id: int):
    images.append(
        {
            "license": images_license,
            "file_name": file_name,
            "coco_url": "",
            "height": height,
            "width": width,
            "date_captured": date_captured,
            "flickr_url": "",
            "id": id,
        }
    )


def add_annotation(image_id: int, category_id: int, bbox: list):
    area = calculate_area(bbox)

    annotations.append(
        {
            "id": len(annotations) + 1,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "iscrowd": iscrowd,
            "attributes": attributes,
        }
    )


def predict_boxes(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_predictions = predict(image)

    boxes = []

    for box in img_predictions.boxes:
        if box.conf < 0.42:
            continue

        data = box.data[0]
        xmin = float(data[0])
        ymin = float(data[1])
        xmax = float(data[2])
        ymax = float(data[3])

        x, y, w, h = xmin, ymin, xmax - xmin, ymax - ymin

        boxes.append(([x, y, w, h], int(box.cls) + 1))

    return boxes


def save_coco_json(save_path: str):
    data = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": "",
        },
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(save_path, "w") as f:
        f.write(json.dumps(data))


def mp4_into_images(
    filepath: str, save_folder_path: str, step: int = 30, enable_predict: bool = False
):

    cap = cv2.VideoCapture(filepath)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        i += 1
        if i % step != 0:
            continue

        # Resize frame to fit 640x640
        # Processed 28800 frames

        # frame = cv2.resize(frame, (640, 640))
        cv2.imwrite(os.path.join(save_folder_path, f"{i}.png"), frame)

        if enable_predict:
            boxes = predict_boxes(frame)
            for box, category_id in boxes:
                add_annotation(i, category_id, box)
            add_image(f"{i}.png", i)

    if enable_predict:
        save_coco_json(os.path.join(save_folder_path, "coco.json"))

    cap.release()


def load_annotations(save_folder_path: str):
    with open(os.path.join(save_folder_path, "coco.json"), "r") as f:
        data = json.load(f)

    global images, annotations
    images = data["images"]
    annotations = data["annotations"]


def extend_annotations(save_folder_path: str):
    load_annotations(save_folder_path)

    for index, image_data in enumerate(images):
        if index % 100 == 0:
            print(f"Processed {index} images")

        if int(image_data["id"]) <= 879:  # type: ignore
            continue  # Skip already processed images

        file_name: str = str(image_data["file_name"])

        image = cv2.imread(os.path.join(save_folder_path, file_name))

        boxes = predict_boxes(image)

        for box, category_id in boxes:
            add_annotation(int(image_data["id"]), category_id, box)  # type: ignore

    save_coco_json(os.path.join(save_folder_path, "coco.json"))
