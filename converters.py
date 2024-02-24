import os

import cv2
import numpy as np


def mp4_into_images(filepath: str, save_folder_path: str):
    cap = cv2.VideoCapture(filepath)
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        i += 1
        if i % 3600 == 0:
            print(f"Processed {i} frames")

            key = input("Continue? (y/n): ")

            if key.lower() == "n":
                break
        # Resize frame to fit 640x640
        frame = cv2.resize(frame, (640, 640))
        cv2.imwrite(os.path.join(save_folder_path, f"{i}.jpg"), frame)

    cap.release()
