import os
from os.path import join
from pathlib import Path
from typing import Any, Callable

import numpy, numpy as np
import SimpleITK as sitk

        
def run_fun_if_not_none(fun: Callable, obj: Any | None = None) -> Any | None:
    """Executes function if object is not None.

    Args:
        fun (callable): 
        obj (Any | None, optional): Defaults to None.

    Returns:
        Any | None: _description_
    """
    if obj is None:
        return None
    return fun(obj)


def copy_if_not_none(obj: Any | None) -> Any | None:
    """Copies the object if not None.

    Args:
        obj (Any | None):

    Returns:
        Any | None:
    """
    fun = lambda x: x.copy()
    return run_fun_if_not_none(fun, obj)


def create_if_not_exists(directory: str):
    """Creates a directory.

    Args:
        directory (str):
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


# Filters
def custom_max_voting_filter(img: numpy.ndarray,
                             radius: int = 3,
                             background_value: int = 0,
                             target_dtype=np.int32) -> numpy.ndarray:
    """Sets all values in radius to the most occuring value.

    Args:
        img (numpy.ndarray): _description_
        radius (int, optional): _description_. Defaults to 3.
        background_value (int, optional): _description_. Defaults to 0.
        target_dtype (_type_, optional): _description_. Defaults to np.int32.

    Returns:
        numpy.ndarray: _description_
    """
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


def get_half_pad_size(value_string: str, max_len: int) -> tuple[int, int]:
    diff = max_len - len(value_string)
    return 1, diff - 1


def derive_output_path(directory: str, fname: str, limit: int = 1000) -> str:
    """Generates a unique output path. If path is already existing,
    adds a counter value until a unique path is found.

    Args:
        directory (str): target directory
        fname (str): target filename
        limit (int, optional): Limit number to prevent endless loops. Defaults to 1000.

    Returns:
        str: Target path
    """
    target_path = join(directory, fname)
    if not os.path.exists(target_path):
        return target_path
    for suffix in range(limit):
        new_target_path = f'{target_path}_{suffix}'
        if not os.path.exists(new_target_path):
            return new_target_path
    return target_path


def derive_unique_directory(directory: str, limit: int = 1000) -> str:
    """Generates a unique output directory. If path is already existing,
    adds a counter value until a unique path is found.

    Args:
        directory (str): target directory
        limit (int, optional): Limit number to prevent endless loops. Defaults to 1000.

    Returns:
        str: Target path
    """
    target_path = directory
    if not os.path.exists(target_path):
        return target_path
    for suffix in range(limit):
        new_target_path = f'{target_path}_{suffix}'
        if not os.path.exists(new_target_path):
            return new_target_path
    return target_path


def simpleitk_to_skimage_interpolation(val: int | str) -> int:
    """Maps the interpolation from SimpleITK to Scikit Image.

    Args:
        val (int): SimpleITK Interpolation mode

    Returns:
        int: Scikit Image Interpolation mode
    """
    if isinstance(val, str):
        if val == 'NN':
            return 0
        elif val == 'LINEAR':
            return 1
    if val == sitk.sitkNearestNeighbor:
        return 0
    elif val == sitk.sitkLinear:
        return 1
    elif val == sitk.sitkBSpline1:
        return 2
    elif val == sitk.sitkBSpline2:
        return 3
    elif val == sitk.sitkBSpline3:
        return 4
    else:
        return 5
