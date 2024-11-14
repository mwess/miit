import os
from os.path import join

import cv2
import numpy, numpy as np
import pydicom
import SimpleITk as sitk
import torch
from torch.autograd import Variable


def load_mri(directory):
    file_id_list = []
    for file_ in os.listdir(directory):
        path = join(directory, file_)
        dc = pydicom.dcmread(path)
        instance_number = int(dc[0x0020, 0x0013].value)
        file_id_list.append((instance_number, file_))
    file_id_list_sorted = sorted(file_id_list, key=lambda x: x[0])
    img_series_reader = sitk.ImageSeriesReader()
    img_series_reader.SetFileNames([join(directory, x[1]) for x in file_id_list_sorted])
    # img_series_reader.
    t2w_img_sitk = img_series_reader.Execute()
    return t2w_img_sitk      


def get_padding_params(img: numpy.array, shape: int) -> tuple[int, int, int, int]:
    pad_x = shape - img.shape[0]
    pad_x_l = pad_x // 2
    pad_x_u = pad_x // 2
    if pad_x % 2 != 0:
        pad_x_u += 1
    pad_y = shape - img.shape[1]
    pad_y_l = pad_y // 2
    pad_y_u = pad_y // 2
    if pad_y % 2 != 0:
        pad_y_u += 1
    return pad_y_l, pad_y_u, pad_x_l, pad_x_u


def scale_image(img, factors: tuple[float, float], interpolation: int = 1):
    w, h = img.shape[:2]
    nw = int(w * factors[0])
    nh = int(h * factors[1])
    return cv2.resize(img, (nh, nw), interpolation=interpolation)


def normalize_image(image, forward=True, mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]):
        mean=torch.FloatTensor(mean).unsqueeze(1).unsqueeze(2)
        std=torch.FloatTensor(std).unsqueeze(1).unsqueeze(2)
        if image.is_cuda:
            mean = mean.cuda()
            std = std.cuda()
        if isinstance(image,torch.Tensor):
            mean = Variable(mean,requires_grad=False)
            std = Variable(std,requires_grad=False)
        return  image

