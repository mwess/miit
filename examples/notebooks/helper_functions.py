from collections import OrderedDict
import glob
import os
from os.path import join, exists
from typing import Optional

import cv2
import numpy, numpy as np
import pandas, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from miit.spatial_data.base_types import Annotation, Image, Pointset
from miit.spatial_data.spatial_omics import Visium, Imzml
from miit.spatial_data import Section
from miit.spatial_data.section import register_to_ref_image
from miit.registerers import ManualAffineRegisterer, get_rotation_matrix_around_center
from miit.registerers.other.msi_to_hist_meta_registerer import MSItoHistMetaRegisterer


pos_to_stain_type = {
    1: 'HES',
    2: 'HE',
    3: 'HES',
    6: 'HES',
    7: 'HES',
    8: 'MTS',
    9: 'IHC',
    10: 'IHC',
    11: 'hes'
}

def load_section(directory, section_number=None):
    """
    Load section from test data.
    """

    # Identify paths
    image_path = glob.glob(join(directory, 'images', '*'))[0]
    mask_path = glob.glob(join(directory, 'masks', '*'))[0]
    lm_path = glob.glob(join(directory, 'landmarks', '*'))[0]
    ann_path = glob.glob(join(directory, 'annotations', '*'))

    # If a section number is given we can assign a staining type as the name.
    if section_number is not None:
        name = pos_to_stain_type[section_number]
    else:
        name = ''
    reference_image = Image.load_from_path(image_path, name=name)

    # Do to some preprocessing mistake, we need to change color channels for sections 2, 8, 9, 10.
    if section_number is not None and section_number in [2, 8, 9, 10]:
        reference_image.data = cv2.cvtColor(reference_image.data, cv2.COLOR_BGR2RGB)
    tissue_mask = Annotation.load_from_path(mask_path, name='tissue_mask')
    tissue_mask.data = tissue_mask.data / tissue_mask.data.max()
    landmarks = Pointset.load_from_path(lm_path, name='landmarks')

    section = Section(reference_image=reference_image, annotations=[tissue_mask, landmarks])
    if len(ann_path) > 0:
        annotation_paths = [x for x in ann_path if x.endswith('.nii.gz')]
        if len(annotation_paths) > 0:
            annotation_path = annotation_paths[0]
            labels_path = [x for x in ann_path if x.endswith('.txt')][0]
            annotation = Annotation.load_from_path(annotation_path, path_to_labels=labels_path, name='tissue_classes', channel_idx=0)
            annotation.data = annotation.data/annotation.data.max()
            section.annotations.append(annotation)
    return section


def load_sections(root_dir, skip_so_data=False):
    sections = OrderedDict()
    sub_dirs = sorted(os.listdir(root_dir), key=lambda x: int(x))
    for sub_dir in sub_dirs:
        sub_path = join(root_dir, sub_dir)
        section = load_section(sub_path, section_number=(int(sub_dir)))
        sections[sub_dir] = section

    # Annotations for section 6 need to cleaned from some faulty pixel information.
    pos_ann = sections['6'].get_annotations_by_names('tissue_classes')
    tissue_mat = sections['6'].get_annotations_by_names('tissue_mask').data.copy()
    new_mat = np.zeros_like(tissue_mat)
    for idx, label in enumerate(pos_ann.labels):
        if label in ['Tissue', 'Stroma']:
            continue
        mat = pos_ann.data[:,:,idx]
        mat[mat > 1] = 1
        new_mat = new_mat + mat
    new_mat[new_mat > 0] = 1
    new_stroma = tissue_mat - new_mat
    new_stroma[new_stroma > 10] = 0
    new_stroma[new_stroma < 0] = 0
    index = pos_ann.labels.index('Tissue')
    pos_ann.data[:,:,index] = sections['6'].get_annotations_by_names('tissue_mask').data.copy()
    index = pos_ann.labels.index('Stroma')
    pos_ann.data[:,:,index] = new_stroma
    data = pos_ann.data.copy()
    data[data > 1] = 1
    pos_ann.data = data


    if not skip_so_data:
        # Imzml
        msi_pos_paths = glob.glob(join(root_dir, '6', 'imzml', '*'))
        msi_pos_imzml_path = [x for x in msi_pos_paths if x.endswith('imzML')][0]
        msi_pos_srd_path = [x for x in msi_pos_paths if x.endswith('srd')][0]
        msi_pos_ibd_path = [x for x in msi_pos_paths if x.endswith('ibd')][0]
        msi_pos_section = sections['6']
        
        msi_pos = Imzml.init_msi_data(msi_pos_imzml_path, name='msi_pos', target_resolution=1)
        msi_pos_pca = msi_pos.get_pca_img()
        warped_msi_pos, _ = register_to_ref_image(msi_pos_section.reference_image.data, msi_pos_pca.data, msi_pos, MSItoHistMetaRegisterer())
        sections['6'].so_data.append(warped_msi_pos)
    
    
        msi_neg_paths = glob.glob(join(root_dir, '7', 'imzml', '*'))
        msi_neg_imzml_path = [x for x in msi_neg_paths if x.endswith('imzML')][0]
        msi_neg_srd_path = [x for x in msi_neg_paths if x.endswith('srd')][0]
        msi_neg_ibd_path = [x for x in msi_neg_paths if x.endswith('ibd')][0]
        msi_neg_section = sections['7']

        msi_neg = Imzml.init_msi_data(msi_neg_imzml_path, name='msi_neg', target_resolution=1)
        msi_neg_pca = msi_neg.get_pca_img()
        warped_msi_neg, _ = register_to_ref_image(msi_neg_section.reference_image.data, msi_neg_pca.data, msi_neg, MSItoHistMetaRegisterer())
        sections['7'].so_data.append(warped_msi_neg)    
    
        st = Visium.from_spcrng(join(root_dir, '2', 'spatial_transcriptomics'))
        warped_st_data, registered_st_image = register_to_ref_image(target_image=sections['2'].reference_image.data,
                                                                    source_image=st.image.data,
                                                                    data=st)
        sections['2'].so_data.append(warped_st_data)
    return sections


def plot_sections(sections, 
                  with_landmarks: bool = True, 
                  image_scale: int = 6,
                  plot_name: bool = True):
    """Plots histology, landmarks and tissue masks for each sections. 

    Args:
        sections (Section): Sections to plot. 
        with_landmarks (bool, optional): If True, plots landmarks in histology images. Defaults to True.
        image_scale (int, optional): Scale of image in plot.
    """
    fig, axs = plt.subplots(len(sections), 2, figsize=(image_scale * 2, image_scale * len(sections)))
    for idx, section in enumerate(sections.values()):
        axs[idx, 0].axis('off')
        axs[idx, 0].imshow(section.reference_image.data)
        axs[idx, 0].set_title('Histology')
        if with_landmarks:
            lms = section.get_annotations_by_names('landmarks')
            if lms is not None:
                axs[idx, 0].plot(lms.data[lms.x_axis], lms.data[lms.y_axis], '.')
        tissue_mask = section.get_annotations_by_names('tissue_mask')
        if tissue_mask is not None:
            mask = tissue_mask.data
            axs[idx, 1].axis('off')
            axs[idx, 1].imshow(mask)
            axs[idx, 1].set_title('Tissue Mask')
        if plot_name:
            axs[idx, 0].set_title(section.reference_image.name)
        


def plot_registration_summary(moving_image: numpy.array,
                              fixed_image: numpy.array,
                              warped_image: numpy.array,
                              moving_lms: Optional[pandas.core.frame.DataFrame] = None,
                              fixed_lms: Optional[pandas.core.frame.DataFrame] = None,
                              warped_lms: Optional[pandas.core.frame.DataFrame] = None,
                              plot_lm_distance: bool = True):
    fig ,axs = plt.subplots(1, 3, figsize=(6 * 3, 6 * 1))
    for ax in axs:
        ax.axis('off')
    axs[0].imshow(moving_image)
    axs[0].set_title('Moving Image')
    if moving_lms is not None:
        axs[0].plot(moving_lms.x, moving_lms.y, 'b.')
    axs[1].imshow(fixed_image)
    axs[1].set_title('Fixed Image')
    if fixed_lms is not None:
        axs[1].plot(fixed_lms.x, fixed_lms.y, 'g.')
    axs[2].imshow(warped_image)
    axs[2].set_title('Warped Image')
    if plot_lm_distance and fixed_lms is not None and warped_lms is not None:
        warped_lms_ = warped_lms.set_index('label', inplace=False)
        fixed_lms_ = fixed_lms.set_index('label', inplace=False)
        joined_lms = warped_lms_.join(fixed_lms_, rsuffix='_fixed')
        for idx, row in joined_lms.iterrows():
            x1 = row['x']
            y1 = row['y']
            x2 = row['x_fixed']
            y2 = row['y_fixed']
            if any(map(lambda x: np.isnan(x) or np.isinf(x), [x1, y1, x2, y2])):
                continue
            axs[2].plot(x1, y1, 'b.')
            axs[2].plot(x2, y2, 'g.')
            x_values = [x1, x2]
            y_values = [y1, y2]
            axs[2].plot(x_values, y_values, 'k', linestyle="-")
    elif warped_lms is not None:
        axs[2].plot(warped_lms.x, warped_lms.y, 'b.')


def make_artificial_integration_section(section: Section):
    """Rotates all annotations of the section by 180 degress except for the so_data.

    Args:
        section (_type_): _description_
    """
    degree = 180
    registerer = ManualAffineRegisterer()
    rotation_matrix = get_rotation_matrix_around_center(section.reference_image.data, degree)
    warped_section = section.apply_transform(registerer, rotation_matrix)
    return warped_section


def get_measurement_dict(df, col1, col2, use_index=True):
    dct = {}
    for idx, row in df.iterrows():
        if use_index:
            dct[idx] = row[col2]
        else:
            dct[row[col1]] = row[col2]
    return dct


def merge_dicts(dict1, dict2):
    new_dict = {}
    for key in dict1:
        if dict1[key] in dict2:
            new_dict[key] = dict2[dict1[key]]
    return new_dict


def get_measurement_matrix_sep(measurement_df, ref_mat, st_table, col):
    local_idx_measurement_dict = get_measurement_dict(measurement_df, '', col, use_index=True)
    intern_idx_local_idx_dict = {}
    for idx, row in st_table.iterrows():
        intern_idx_local_idx_dict[int(row['int_idx'])] = idx
    merged_dict = merge_dicts(intern_idx_local_idx_dict, local_idx_measurement_dict)
    indexer = np.array([merged_dict.get(i, 0) for i in range(ref_mat.min(), ref_mat.max() + 1)])
    measurement_mat = indexer[(ref_mat - ref_mat.min())]
    return measurement_mat


def get_measurement_matrix_2(measurement_df, ref_mat, table, col):
    st_table = table.loc[measurement_df.index.to_list()]
    return get_measurement_matrix_sep(measurement_df, ref_mat, st_table, col)


def get_spot_coloring_img(df, 
                          ref_mat, 
                          table, 
                          col='unified_hp_class', 
                          background_color='white',
                          stroma='stroma',
                          gland='gland'):
    """Utility function for plotting gland/stroma distribution.

    Args:
        df (_type_): _description_
        ref_mat (_type_): _description_
        table (_type_): _description_
        col (str, optional): _description_. Defaults to 'unified_hp_class'.
        background_color (str, optional): _description_. Defaults to 'white'.
        stroma (str, optional): _description_. Defaults to 'stroma'.
        gland (str, optional): _description_. Defaults to 'gland'.

    Returns:
        _type_: _description_
    """
    stroma_color = sns.color_palette()[0]
    gland_color = sns.color_palette()[1]
    spot_class_mat = get_measurement_matrix_2(df, ref_mat, table, col)
    smat = np.zeros((spot_class_mat.shape[0], spot_class_mat.shape[1], 3), dtype=np.float32)
    if background_color == 'white':
        smat += 1
    for i in range(spot_class_mat.shape[0]):
        for j in range(spot_class_mat.shape[1]):
            val = spot_class_mat[i,j]
            if val == gland:
                smat[i,j] = gland_color
            elif val == stroma:
                smat[i,j] = stroma_color
    smat = (smat * 255).astype(np.uint8)
    return smat

    
def min_max_normalize(pd_series):
    min_ = pd_series.min()
    max_ = pd_series.max()
    return (pd_series - min_)/(max_ - min_)

    
def normalize_z(arr):
    return (arr - arr.mean())/arr.std()