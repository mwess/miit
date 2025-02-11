import cv2
import numpy, numpy as np


def remove_padding(image: numpy.ndarray, 
                   padding: tuple[int, int, int, int]) -> numpy.ndarray:
    """Removes padding from image.

    Args:
        image (numpy.ndarray): 
        padding (Tuple[int, int, int, int]): 

    Returns:
        numpy.ndarray: 
    """
    left, right, top, bottom = padding
    bottom_idx = -bottom if bottom != 0 else image.shape[0]
    right_idx = -right if right != 0 else image.shape[1]
    return image[top:bottom_idx, left:right_idx]


def get_symmetric_padding(img1: numpy.ndarray, img2: numpy.ndarray) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    max_size = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    padding_img1 = get_padding_params(img1, max_size)
    padding_img2 = get_padding_params(img2, max_size)
    return padding_img1, padding_img2


def pad_asym(image: numpy.ndarray, padding: tuple[int, int, int, int], constant_values: int = 0) -> numpy.ndarray:
    """Applies padding to image.

    Args:
        image (numpy.ndarray): 
        padding (Tuple[int, int, int, int]): padding is organized as top, bottom, left, right
        constant_values (int, optional): Defaults to 0.

    Returns:
        numpy.ndarray: 
    """
    left, right, top, bottom = padding
    if len(image.shape) == 2:
        image = np.pad(image, ((top, bottom), (left, right)), constant_values=constant_values)
    else:
        # Assume 3 dimensions
        image = np.pad(image, ((top, bottom), (left, right), (0, 0)), constant_values=constant_values)
    return image


def get_padding_params(img: numpy.ndarray, shape: int) -> tuple[int, int, int, int]:
    """Computes padding parameters to pad image to given shape.

    Args:
        img (numpy.ndarray): 
        shape (int): 

    Returns:
        tuple[int, int, int, int]: 
    """
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


def apply_mask(image: numpy.ndarray, mask: numpy.ndarray) -> numpy.ndarray:
    """Applies mask to image

    Args:
        image (numpy.ndarray): 
        mask (numpy.ndarray): 

    Returns:
        numpy.ndarray: 
    """
    if len(image.shape) == 3:
        mask = np.moveaxis(np.expand_dims(mask, 0), 0, -1)
    return image * mask


def write_affine_to_file(mat: numpy.ndarray, path: str):
    """Writes affine matrix to a text file.
    
    Args:
        mat (numpy.ndarray):
        path (str):
        
    """
    with open(path, 'w') as f:
        output_str = f"""{mat[0,0]} {mat[0,1]} {mat[0,2]}\n{mat[1,0]} {mat[1,1]} {mat[1,2]}\n0.0 0.0 1.0"""
        f.write(output_str)


def pad_to_image(source: numpy.ndarray, target: numpy.ndarray, background_value: int = 0) -> numpy.ndarray:
    """
    Pads source image symmetrically to the same shape as target.

    Args:
        source (numpy.ndarray): 
        target (numpy.ndarray): 
        background_value (int, optional): Defaults to 0.

    Returns:
        numpy.ndarray: 
    """
    x_diff = target.shape[0] - source.shape[0]
    y_diff = target.shape[1] - source.shape[1]
    pad_x_left = x_diff // 2
    pad_x_right = pad_x_left
    if x_diff % 2 == 1:  
        pad_x_right += 1        
    pad_y_left = y_diff // 2
    pad_y_right = pad_y_left
    if y_diff % 2 == 1:
        pad_y_right += 1
    padding = {
        'pad_y_right': pad_y_right,
        'pad_y_left': pad_y_left,
        'pad_x_left': pad_x_left,
        'pad_x_right': pad_x_right
    }
    return cv2.copyMakeBorder(source, pad_x_right, pad_x_left, pad_y_left, pad_y_right, cv2.BORDER_CONSTANT, background_value), padding