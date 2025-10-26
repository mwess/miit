import cv2
import numpy, numpy as np

def make_circle_image_multichannel(draw_c1: bool = True,
                                   draw_c2: bool = True) -> numpy.ndarray :
    shape = (500, 500)
    base_img = np.zeros(shape, dtype=np.int32)

    c1 = (200, 200)
    r1 = 70
    v1 = 1

    c2 = (400, 400)
    r2 = 50
    v2 = 2

    if draw_c1:
        base_img = cv2.circle(base_img, c1, r1, v1, -1)

    if draw_c2:
        base_img = cv2.circle(base_img, c2, r2, v2, -1)

    return base_img


def make_circle_image_singlechannel(draw_c1: bool = True,
                                    draw_c2: bool = True):
    label_counter = 0
    if draw_c1:
        label_counter += 1
    if draw_c2:
        label_counter += 1
    shape = (500, 500)
    base_img = np.zeros(shape, dtype=np.int32)

    c1 = (200, 200)
    r1 = 70
    v1 = 1

    c2 = (400, 400)
    r2 = 50
    v2 = 2

    circle_list = []
    if draw_c1:
        circle_list.append((c1, r1, v1))
    if draw_c2:
        circle_list.append((c2, r2, v2))

    circles = []
    for _, (c, r, v) in enumerate(circle_list):
        base_img = cv2.circle(base_img.copy(), c, r, v, -1)
        circles.append(base_img)
    img = np.dstack(circles)
    return img


def get_circle_image_labels() -> list[str]:
    return ['label1', 'label2']