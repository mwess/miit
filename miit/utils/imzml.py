
from typing import Optional, List, Tuple, Dict
from miit.registerers.base_registerer import Registerer
from miit.registerers.nifty_reg import NiftyRegWrapper

import cv2
import numpy, numpy as np
import SimpleITK, SimpleITK as sitk
from sklearn.decomposition import PCA
import pyimzml
from greedyfhist.segmentation import load_yolo_segmentation

from .image_utils import (
    get_symmetric_padding, 
    pad_asym,
    remove_padding
)


# TODO: Remove that function
def do_msi_registration(histology_image: numpy.ndarray, 
                        ref_mat: numpy.ndarray, 
                        spec_to_ref_map: Dict, 
                        msi: pyimzml.ImzMLParser.ImzMLParser, 
                        reg_img:Optional[numpy.ndarray]=None, 
                        additional_images:Optional[List[numpy.array]]=None,
                        registerer: Optional[Registerer] = None) -> Tuple[numpy.ndarray, numpy.ndarray, List[numpy.ndarray]]:
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
    


# TODO: Remove that function
def post_registration_transforms(warped_images: List[numpy.array], processing_dict: Dict) -> List[numpy.array]:
    global_padding = processing_dict['global_padding']
    unpadded_images = []
    for warped_image_np in warped_images:
        # warped_image_np = sitk.GetArrayFromImage(warped_image)
        warped_image_np = warped_image_np[global_padding:-global_padding,global_padding:-global_padding]
        warped_image_np = remove_padding(warped_image_np, processing_dict['fix_sym_pad'])
        unpadded_images.append(warped_image_np)
    return unpadded_images


def get_pca_img(msi: pyimzml.ImzMLParser.ImzMLParser, 
                ref_mat: numpy.ndarray, 
                spec_to_ref_map: dict, 
                mz_threshold: Optional[float]=None) -> numpy.ndarray:
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


# TODO: Remove that function
def preprocess_for_registration(fixed_image: numpy.ndarray, 
                                moving_image: numpy.ndarray, 
                                ref_mat: numpy.ndarray,
                                additional_image_datas:Optional[List[numpy.array]]=None,
                                padding: int =100) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, List[numpy.ndarray], dict]:
    fixed_image_np, process_dict = preprocess_histology(fixed_image, moving_image)
    # Add global padding
    # First do asym padding
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



def preprocess_histology(hist_img: numpy.ndarray, 
                         msi_img: numpy.ndarray,
                         hist_img_mask: Optional[numpy.ndarray] = None) -> Tuple[numpy.ndarray, dict]:
    """
    Preprocessing steps: Remove background noise, pad to optimally match the shape of the moving image.
    """
    # Remove noise
    if hist_img_mask is None:
        segmentation_fun = load_yolo_segmentation()
        hist_img_mask = segmentation_fun(hist_img)
    hist_gray = cv2.cvtColor(hist_img, cv2.COLOR_RGB2GRAY)
    image_dict = {}
    hist_gray = hist_gray * hist_img_mask
    fix_pad, mov_pad = get_symmetric_padding(hist_gray, msi_img)
    image_dict['mov_sym_pad'] = mov_pad
    image_dict['fix_sym_pad'] = fix_pad
    return hist_gray, image_dict


def get_mode(msi, use_auto=False):
    if use_auto:
        return 'auto'
    is_continuous = msi.metadata.file_description.param_by_name['continuous']
    if is_continuous or is_continuous is None:
        return 'continuous'
    else:
        return 'processed'


def get_spec_type(msi, default_spec_type = 'centroid'):
    rpg = msi.metadata.referenceable_param_groups['spectrum']
    if rpg is None:
        return default_spec_type
    spec_type = rpg.param_by_accession.get('MS:1000127', None)
    if spec_type is not None and spec_type:
        return 'centroid'
    else:
        return 'profile'


def get_scan_direction(msi, default_scan_direction='top_down'):

    # We take the first key.
    key = list(msi.metadata.scan_settings.keys())[0]
    scan_settings_params = msi.metadata.scan_settings[key]
    scan_directions = {
        "1000400": "bottom_up",
        "1000402": "left_right",
        "1000403": "right_left",
        "1000401": "top_down"
    }
    for key in scan_settings_params.param_by_accession:
        if scan_settings_params.param_by_accession[key] == True:
            accession_number = key.split(':')[-1]
            if accession_number in scan_directions:
                return scan_directions[accession_number]
    return default_scan_direction


def get_line_scan_direction(msi, default_scan_direction='line_left_right'):

    # We take the first key.
    key = list(msi.metadata.scan_settings.keys())[0]
    scan_settings_params = msi.metadata.scan_settings[key]
    line_scan_directions = {
        "1000492": "line_bottom_up",
        "1000491": "line_left_right",
        "1000490": "line_right_left",
        "1000493": "line_top_down"
    }
    for key in scan_settings_params.param_by_accession:
        if scan_settings_params.param_by_accession[key] == True:
            accession_number = key.split(':')[-1]
            if accession_number in line_scan_directions:
                return line_scan_directions[accession_number]
    return default_scan_direction


def get_scan_pattern(msi, default_scan_pattern='one_way'):

    # We take the first key.
    key = list(msi.metadata.scan_settings.keys())[0]
    scan_settings_params = msi.metadata.scan_settings[key]
    scan_patterns ={
        "1000410": "meandering",
        "1000411": "one_way",
        "1000412": "random_access",
    }    
    for key in scan_settings_params.param_by_accession:
        if scan_settings_params.param_by_accession[key] == True:
            accession_number = key.split(':')[-1]
            if accession_number in scan_patterns:
                return scan_patterns[accession_number]
    return default_scan_pattern 
    

def get_scan_type(msi, default_scan_type='horizontal_line'):
    key = list(msi.metadata.scan_settings.keys())[0]
    scan_settings_params = msi.metadata.scan_settings[key]
    scan_types = {
        "1000480": "horizontal_line",
        "1000481": "vertical_line"
    }
    for key in scan_settings_params.param_by_accession:
        if scan_settings_params.param_by_accession[key] == True:
            accession_number = key.split(':')[-1]
            if accession_number in scan_types:
                return scan_types[accession_number]
    return default_scan_type 
    
