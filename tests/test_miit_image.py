import tempfile

import cv2
import numpy, numpy as np

from miit.spatial_data import Image


def test_make_image_from_np_eq_img():
    np_img = np.zeros((800, 600, 3), dtype=np.float32)
    img = Image(data=np_img)
    assert (img.data == np_img).all()


def test_make_image_from_np_eq_name():
    name = 'np_img'
    img = Image(name=name)
    assert img.name == name


def test_make_image_from_file_eq_img():
    shape = (800, 600, 3)
    np_img = np.zeros(shape, dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        img_path = f.name
        cv2.imwrite(img_path, np_img)
        img = Image.load_from_path(img_path)
    assert shape == img.data.shape