import cv2
import numpy, numpy as np
from sklearn.decomposition import PCA
import pyimzml, pyimzml.ImzMLParser
from greedyfhist.segmentation import load_yolo_segmentation

from .image_utils import (
    get_symmetric_padding, 
    pad_asym,
)


def get_pca_img(msi: pyimzml.ImzMLParser.ImzMLParser, 
                ref_mat: numpy.ndarray, 
                spec_to_ref_map: dict, 
                mz_threshold: float | None = None) -> numpy.ndarray:
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


def preprocess_histology(hist_img: numpy.ndarray, 
                         msi_img: numpy.ndarray,
                         hist_img_mask: numpy.ndarray | None = None) -> tuple[numpy.ndarray, dict]:
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


def get_mode(msi, use_auto=False) -> str:
    if use_auto:
        return 'auto'
    is_continuous = msi.metadata.file_description.param_by_name['continuous']
    if is_continuous or is_continuous is None:
        return 'continuous'
    else:
        return 'processed'


def get_spec_type(msi, default_spec_type = 'centroid') -> str:
    rpg = msi.metadata.referenceable_param_groups['spectrum']
    if rpg is None:
        return default_spec_type
    spec_type = rpg.param_by_accession.get('MS:1000127', None)
    if spec_type is not None and spec_type:
        return 'centroid'
    else:
        return 'profile'


def get_scan_direction(msi, default_scan_direction='top_down') -> str:

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


def get_line_scan_direction(msi, default_scan_direction='line_left_right') -> str:

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


def get_scan_pattern(msi, default_scan_pattern='one_way') -> str:

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
    

def get_scan_type(msi, default_scan_type='horizontal_line') -> str:
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
    
