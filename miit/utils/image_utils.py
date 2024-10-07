from typing import Any, Dict, List, Tuple

import numpy, numpy as np

from miit.spatial_data.section import Section


def get_symmetric_padding(img1: numpy.array, img2: numpy.array) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    max_size = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    padding_img1 = get_padding_params(img1, max_size)
    padding_img2 = get_padding_params(img2, max_size)
    return padding_img1, padding_img2


def get_padding_params(img: numpy.array, shape: int) -> Tuple[int, int, int, int]:
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


def get_symmetric_padding_for_sections(sections: List[Section]) -> Dict[Any, Tuple[int, int, int, int]]:
    paddings = {}
    max_size = get_max_size_from_sections(sections)
    for section in sections:
        paddings[section._id] = get_padding_params(section.reference_image.data, max_size)
    return paddings


def get_section_max_size(section: Section) -> int:
    return max(section.reference_image.data.shape[0], section.reference_image.data.shape[1])


def get_max_size_from_sections(sections: List[Section]) -> int:
    max_size = -1
    for section in sections:
        max_size = max(max_size, get_section_max_size(section))
    return max_size


def apply_mask(image: numpy.array, mask: numpy.array) -> numpy.array:
    if len(image.shape) == 3:
        mask = np.moveaxis(np.expand_dims(mask, 0), 0, -1)
    return image * mask


def write_affine_to_file(mat: numpy.ndarray, path: str):
    with open(path, 'w') as f:
        output_str = f"""{mat[0,0]} {mat[0,1]} {mat[0,2]}\n{mat[1,0]} {mat[1,1]} {mat[1,2]}\n0.0 0.0 1.0"""
        f.write(output_str)
