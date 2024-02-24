import time

import cv2
import numpy as np
from mss import mss

bounding_box = {"top": 100, "left": 0, "width": 640, "height": 480}

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


@add_print_fps
def draw():
    sct_img = sct.grab(bounding_box)
    cv2.imshow("screen", np.array(sct_img))


def run():
    stop = False

    while not stop:
        draw()

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            stop = True

    destroy()


if __name__ == "__main__":
    run()
