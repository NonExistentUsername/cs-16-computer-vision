import random
import threading
import time

import cv2
import keyboard  # type: ignore
import numpy as np
import pyautogui  # type: ignore
from mss import mss

from predictions import model, predict

bounding_box = {"top": 60, "left": 0, "width": 1600, "height": 600}

cv2.namedWindow("screen", cv2.WINDOW_NORMAL)

sct = mss()

auto_shoot = False

conf_limit = 0.45


def destroy():
    cv2.destroyAllWindows()
    sct.close()


def add_print_fps(draw_func):
    def wrapper():
        start = time.time()
        draw_func()
        delta = time.time() - start
        print(f"FPS: {1 / delta:.2f}\nTime: {delta * 1000:.2f} ms")

    return wrapper


class_names = {0: "ct_body", 1: "ct_head", 2: "t_body", 3: "t_head"}
print(model.names)
colors = [(0, 0, 255), (126, 126, 255), (255, 126, 126), (255, 255, 0)]


def draw_box(img, data):
    xmin = int(data[0])
    ymin = int(data[1])
    xmax = int(data[2])
    ymax = int(data[3])

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[1], 2)


def draw_predictions(img, prediction):
    for box in prediction.boxes:
        if box.conf < conf_limit:
            continue

        data = box.data[0]
        xmin = int(data[0])
        ymin = int(data[1])
        xmax = int(data[2])
        ymax = int(data[3])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[int(box.cls)], 2)


img_predictions = None


def predict_in_background():
    global img_predictions
    while True:
        sct_img = sct.grab(bounding_box)
        sct_img = np.array(sct_img)
        sct_img = cv2.resize(sct_img, (640, 640))
        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGR2RGB)
        img_predictions = predict(sct_img)


def start_predictions():
    import threading

    t = threading.Thread(target=predict_in_background, daemon=True)
    t.start()


# @add_print_fps
def draw():
    global img_predictions
    sct_img = sct.grab(bounding_box)
    sct_img = np.array(sct_img)
    sct_img = cv2.resize(sct_img, (640, 640))
    draw_predictions(sct_img, img_predictions)
    cv2.imshow("screen", sct_img)


def window_position_to_real_position(x, y):
    window_x, window_y = bounding_box["left"], bounding_box["top"]
    window_width, window_height = bounding_box["width"], bounding_box["height"]

    x = x / 640 * window_width
    y = y / 640 * window_height

    x += window_x
    y += window_y

    return x, y


def move_mouse(x, y):
    x, y = window_position_to_real_position(x, y)
    pyautogui.moveTo(x, y, duration=0.01)
    # pyautogui.click()


def get_mouse_position():
    return 640 / 2, 640 / 2


def get_closest_box(x, y):
    closest_box = None
    is_head = False
    closest_distance = float("inf")

    for box in img_predictions.boxes:
        if box.conf < conf_limit:
            continue

        if is_head and (int(box.cls) == 0 or int(box.cls) == 2):
            continue

        if is_head == False and (int(box.cls) == 1 or int(box.cls) == 3):
            is_head = True
            closest_box = box
            continue

        data = box.data[0]
        xmin = int(data[0])
        ymin = int(data[1])
        xmax = int(data[2])
        ymax = int(data[3])

        distance = (x - (xmin + xmax) / 2) ** 2 + (y - (ymin + ymax) / 2) ** 2

        if distance < closest_distance:
            closest_distance = distance
            closest_box = box

    max_distance = 60**2

    if closest_distance > max_distance:
        return None

    return closest_box


def move_mouse_to_closest_box(x, y):
    box = get_closest_box(x, y)

    if box is not None:
        data = box.data[0]
        xmin = int(data[0])
        ymin = int(data[1])
        xmax = int(data[2])
        ymax = int(data[3])

        x = (xmin + xmax) / 2
        y = (ymin + ymax) / 2

        move_mouse(x, y)


def move_mouse_to_closest_box_in_background():
    while True:
        if auto_shoot:
            time.sleep(0.1)


mouse_modifier = False


def run():
    global auto_shoot
    global mouse_modifier

    stop = False

    for _ in range(3):
        time.sleep(0.1)
        start_predictions()

    while img_predictions is None:
        print("Waiting for predictions")
        time.sleep(0.1)

    last_shoot = time.time()
    last_auto_shoot_change = time.time()

    while not stop:
        draw()

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            stop = True

        if (
            keyboard.is_pressed("command")
            and time.time() - last_auto_shoot_change > 0.5
        ):
            auto_shoot = not auto_shoot

            if auto_shoot:
                print("Auto shoot enabled")
            else:
                print("Auto shoot disabled")

            last_auto_shoot_change = time.time()

        if auto_shoot and time.time() - last_shoot > 0.1:
            t = threading.Thread(
                target=lambda: move_mouse_to_closest_box(*get_mouse_position()),
                daemon=True,
            )
            t.start()
            last_shoot = time.time()

    destroy()


if __name__ == "__main__":
    run()
