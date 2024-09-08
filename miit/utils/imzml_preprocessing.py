
from typing import Optional, List, Tuple, Dict
from miit.registerers.base_registerer import Registerer
from miit.registerers.nifity_reg import NiftyRegWrapper

import cv2
import numpy
import numpy as np
import SimpleITK as sitk

from greedyfhist.segmentation import load_yolo_segmentation


import numpy as np
from sklearn.decomposition import PCA


def do_msi_registration(histology_image: numpy.array, 
                        ref_mat: numpy.array, 
                        spec_to_ref_map: Dict, 
                        msi, 
                        reg_img:numpy.array=None, 
                        additional_images:Optional[List[numpy.array]]=None,
                        registerer: Optional[Registerer] = None):
    if additional_images is None:
        additional_images = []
    if reg_img is None:
        reg_img = get_pca_img(msi, ref_mat, spec_to_ref_map)
    (fixed_image, 
     moving_image, 
     proc_ref_mat,
     proc_add_imgs,
     padding_dict
    ) = preprocess_for_registration(histology_image, reg_img, ref_mat, additional_image_datas=additional_images)

    if registerer is None:
        registerer = NiftyRegWrapper.load_from_config({})
    registration_result = registerer.register_images(moving_image, fixed_image)
    warped_image = registerer.transform_image(moving_image, registration_result, 'LINEAR')
    warped_ref_mat = registerer.transform_image(proc_ref_mat, registration_result, 'NN')
    warped_add_imgs = []
    for img in proc_add_imgs:
        warped_add_img = registerer.transform_image(img, registration_result, 'NN')
        warped_add_imgs.append(warped_add_img)
    unpadded_images = post_registration_transforms([warped_image, warped_ref_mat] + warped_add_imgs, padding_dict)
    unpadded_warped_img = unpadded_images[0]
    unpadded_ref_mat = unpadded_images[1]
    unpadded_add_imgs = unpadded_images[2:]
    return unpadded_warped_img, unpadded_ref_mat, unpadded_add_imgs
    

def post_registration_transforms(warped_images: List[numpy.array], processing_dict: Dict) -> List[numpy.array]:
    global_padding = processing_dict['global_padding']
    unpadded_images = []
    for warped_image_np in warped_images:
        # warped_image_np = sitk.GetArrayFromImage(warped_image)
        warped_image_np = warped_image_np[global_padding:-global_padding,global_padding:-global_padding]
        warped_image_np = remove_padding(warped_image_np, processing_dict['fix_sym_pad'])
        unpadded_images.append(warped_image_np)
    return unpadded_images


def remove_padding(image, padding: Tuple[int, int, int, int]):
    left, right, top, bottom = padding
    bottom_idx = -bottom if bottom != 0 else image.shape[0]
    right_idx = -right if right != 0 else image.shape[1]
    return image[top:bottom_idx, left:right_idx]


def resize_image_simple_sitk(image, res, out_type=np.float32):
    img_np = sitk.GetArrayFromImage(image)
    new_img_np = cv2.resize(img_np.astype(np.float32), (res[1], res[0]), 0, 0, interpolation=cv2.INTER_NEAREST)
    return sitk.GetImageFromArray(new_img_np.astype(np.float32))


def get_pca_img(msi, ref_mat, spec_to_ref_map, mz_threshold=None):
    # Collect all spectra
    msi_data = []
    for idx, _ in enumerate(msi.coordinates):
        _, ints = msi.getspectrum(idx)
        msi_data.append(ints)

    msi_mat = np.array(msi_data)
    if mz_threshold is not None:
        msi_mat[msi_mat <= mz_threshold] = 0   

    # Compute PCA space
    pca = PCA(n_components=1)
    pca.fit(msi_mat)
    reduced_mz = pca.transform(msi_mat)
    reduced_mz = reduced_mz.squeeze()

    # Map PCA values to image space
    map_mz_dict = {}
    for idx, val in enumerate(reduced_mz):
        ref_idx = spec_to_ref_map[idx]
        map_mz_dict[ref_idx] = val
    indexer = np.array([map_mz_dict.get(i, 0) for i in range(ref_mat.min(), ref_mat.max() + 1)])
    pca_mz_mat = indexer[(ref_mat - ref_mat.min())]
    return pca_mz_mat

    
def pad_asym(image, padding: Tuple[int, int, int, int], constant_values: int = 0):
        left, right, top, bottom = padding
        if len(image.shape) == 2:
            image = np.pad(image, ((top, bottom), (left, right)), constant_values=constant_values)
        else:
            # Assume 3 dimensions
            image = np.pad(image, ((top, bottom), (left, right), (0, 0)), constant_values=constant_values)
        return image


def preprocess_for_registration(fixed_image, 
                                moving_image, 
                                ref_mat,
                                additional_image_datas:Optional[List[numpy.array]]=None,
                                padding=100):
    fixed_image_np, process_dict = preprocess_histology(fixed_image, moving_image, background_value=0)
    # Add global padding
    # First do aym padding
    fixed_image_padding = pad_asym(fixed_image_np, process_dict['fix_sym_pad'])
    fixed_image_padding = np.pad(fixed_image_padding, ((padding, padding),(padding, padding)))
    moving_image_padding = pad_asym(moving_image, process_dict['mov_sym_pad'])
    moving_image_padding = np.pad(moving_image_padding, ((padding, padding),(padding, padding)))
    # Here do first asym data
    ref_mat_padding = pad_asym(ref_mat, process_dict['mov_sym_pad'])
    ref_mat_padding = np.pad(ref_mat_padding, ((padding, padding), (padding, padding)))
    if additional_image_datas is not None:
        padded_additional_image_datas = []
        for image_data in additional_image_datas:
            padded_image_data = pad_asym(image_data, process_dict['mov_sym_pad'])
            padded_image_data = np.pad(padded_image_data, ((padding, padding), (padding, padding)))
            padded_additional_image_datas.append(padded_image_data)
    else:
        padded_additional_image_datas = None
    process_dict['global_padding'] = padding
    return fixed_image_padding, moving_image_padding, ref_mat_padding, padded_additional_image_datas, process_dict


def preprocess_histology(hist_img, moving_img, background_value=0, background_cutoff=215):
    """
    Preprocessing steps: Remove background noise, pad to optimally match the shape of the moving image.
    """
    # Remove noise
    segmentation_fun = load_yolo_segmentation()
    mask = segmentation_fun(hist_img)
    hist_gray = cv2.cvtColor(hist_img, cv2.COLOR_RGB2GRAY)
    image_dict = {'segmentation_mask': mask}
    image_dict['gray'] = hist_gray
    hist_gray = hist_gray * mask
    fix_pad, mov_pad = get_symmetric_padding(hist_gray, moving_img)
    image_dict['mov_sym_pad'] = mov_pad
    image_dict['fix_sym_pad'] = fix_pad
    # hist_gray, padding = pad_to_image(hist_gray, moving_img)
    # image_dict['padding'] = padding
    return hist_gray, image_dict


def get_symmetric_padding(img1: numpy.array, img2: numpy.array):
    max_size = max(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    padding_img1 = get_padding_params(img1, max_size)
    padding_img2 = get_padding_params(img2, max_size)
    return padding_img1, padding_img2


def get_padding_params(img: numpy.array, shape: int):
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


def pad_to_image(source, target, background_value=0):
    # Assumes that 
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