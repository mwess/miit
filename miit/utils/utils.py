# from dataclasses import Protocol

import math
import os
from pathlib import Path
from typing import TypeVar, Dict, List, Tuple

import numpy, numpy as np

# T = TypeVar('T')

# class IsCopyable(Protocol[T]):
    
#     def copy(self: T) -> T:
#         ...
        
def run_fun_if_not_none(fun, obj=None):
    if obj is None:
        return None
    return fun(obj)


# def copy_if_not_none(obj: IsCopyable):
def copy_if_not_none(obj):
    fun = lambda x: x.copy()
    return run_fun_if_not_none(fun, obj)


def create_if_not_exists(directory: str):
    Path(directory).mkdir(parents=True, exist_ok=True)


# Filters
def custom_max_voting_filter(img: numpy.array,
                             radius: int = 3,
                             background_value: int = 0,
                             target_dtype=np.int32):
    filtered_image = np.zeros(img.shape, dtype=target_dtype)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i,j] == background_value:
                continue
            xmin = max(i - radius, 0)
            xmax = min(i + radius, img.shape[0]-1)
            ymin = max(j - radius, 0)
            ymax = min(j + radius, img.shape[1]-1)
            window = img[xmin:xmax, ymin:ymax]
            uniques, counts = np.unique(window, return_counts=True)
            max_idx = np.argmax(counts)
            filtered_image[i, j] = uniques[max_idx].astype(target_dtype)
    return filtered_image


def clean_configs(config: Dict) -> Dict:
    for section in config['sections']:
        if 'molecular_imaging_data' in section:
            del section['molecular_imaging_data']
    return config


def filter_node_ids(config: Dict, id_list: List) -> Dict:
    keep_sections = []
    for section in config['sections']:
        # print(section['id'])
        if section['id'] in id_list:
            keep_sections.append(section)
    config['sections'] = keep_sections
    return config


def get_half_pad_size(value_string: str, max_len: int) -> Tuple[int, int]:
    diff = max_len - len(value_string)
    if diff % 2 == 0:
        l_pad = diff / 2
        r_pad = diff / 2
    else:
        l_pad = math.floor(diff / 2)
        r_pad = math.ceil(diff / 2)
    return 1, diff - 1