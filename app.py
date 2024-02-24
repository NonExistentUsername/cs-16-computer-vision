import random
import time

import cv2
import numpy as np
from mss import mss

from predictions import model, predict

bounding_box = {"top": 200, "left": 0, "width": 640, "height": 640}

sct = mss()


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


class_names = model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]


def draw_predictions(img, prediction):
    for box in prediction.boxes:
        data = box.data[0]
        xmin = int(data[0])
        ymin = int(data[1])
        xmax = int(data[2])
        ymax = int(data[3])

        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), colors[int(box.cls)], 2)


img_predictions = None


@add_print_fps
def draw():
    global img_predictions
    sct_img = sct.grab(bounding_box)
    sct_img = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGR2RGB)
    img_predictions = predict(sct_img)
    draw_predictions(sct_img, img_predictions)
    cv2.imshow("screen", cv2.cvtColor(sct_img, cv2.COLOR_RGB2BGR))


def run():
    stop = False

    while not stop:
        draw()

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            stop = True

    destroy()


if __name__ == "__main__":
    run()
