import time
from multiprocessing import Manager, Pool, Process

import cv2
import numpy as np
from mss import mss

from predictions import model, predict

# import Value from multiprocessing


bounding_box = {"top": 60, "left": 0, "width": 1600, "height": 1200}
# bounding_box = {"top": 60, "left": 0, "width": 2389, "height": 1024}

cv2.namedWindow("screen", cv2.WINDOW_NORMAL)

image_size = 1024
image_sizes = (image_size, image_size)

sct = mss()

auto_shoot = False
conf_limit = 0.5

# image_predictions: deque = deque(maxlen=1)


def destroy():
    cv2.destroyAllWindows()
    sct.close()


def add_print_fps(draw_func):
    def wrapper(*args, **kwargs):
        start = time.time()
        sct_img, image_predictions = draw_func(*args, **kwargs)
        delta = time.time() - start
        cv2.putText(
            sct_img,
            f"FPS: {1 / delta:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return sct_img, image_predictions

    return wrapper


class_names = {0: "ct_body", 1: "ct_head", 2: "t_body", 3: "t_head"}
colors = [(0, 0, 255), (126, 126, 255), (255, 126, 126), (255, 255, 0)]


def draw_box(img, data):
    xmin = int(data[0])
    ymin = int(data[1])
    xmax = int(data[2])
    ymax = int(data[3])

    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[1], 2)


def draw_predictions(img, prediction):
    for box in prediction:
        xmin, ymin, xmax, ymax, color = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[color], 2)


def process_predictions(predictions):
    boxes = []
    for box in predictions.boxes:
        if box.conf < conf_limit:
            continue

        data = box.data[0]
        xmin = int(data[0])
        ymin = int(data[1])
        xmax = int(data[2])
        ymax = int(data[3])

        boxes.append((xmin, ymin, xmax, ymax, int(box.cls)))

    return boxes


def predict_in_background(image_predictions):
    while True:
        sct_img = sct.grab(bounding_box)
        sct_img = np.array(sct_img)
        sct_img = cv2.resize(sct_img, image_sizes)
        sct_img = cv2.cvtColor(sct_img, cv2.COLOR_BGR2RGB)
        predictions = predict(sct_img)
        processed_predictions = process_predictions(predictions)
        image_predictions.put(processed_predictions)


@add_print_fps
def draw(image_predictions):
    image_predictions = image_predictions.get()
    sct_img = sct.grab(bounding_box)
    sct_img = np.array(sct_img)
    sct_img = cv2.resize(sct_img, image_sizes)
    draw_predictions(sct_img, image_predictions)
    return sct_img, image_predictions


def window_position_to_real_position(x, y):
    window_x, window_y = bounding_box["left"], bounding_box["top"]
    window_width, window_height = bounding_box["width"], bounding_box["height"]

    x = x / image_size * window_width
    y = y / image_size * window_height

    x += window_x
    y += window_y

    return x, y


def get_mouse_position():
    return 1024 / 2, 1024 / 2


def move_mouse_to_closest_box_in_background():
    while True:
        if auto_shoot:
            time.sleep(0.1)


mouse_modifier = False


def run():
    global auto_shoot
    global mouse_modifier

    stop = False

    # Create a manager for the deque and start a server
    with Manager() as manager:
        image_predictions = manager.Queue(maxsize=4)

        # with manager.Pool(1) as p:
        # p.apply_async(predict_in_background, args=[image_predictions])
        for _ in range(2):
            p = Process(target=predict_in_background, args=[image_predictions])
            p.start()

        last_shoot = time.time()
        last_auto_shoot_change = time.time()

        while not stop:
            sct_img, boxes = draw(image_predictions)
            cv2.imshow("screen", sct_img)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                stop = True

        destroy()


if __name__ == "__main__":
    run()
