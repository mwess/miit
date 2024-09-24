from collections import OrderedDict
import glob
import os
from os.path import join, exists
from typing import Optional

import numpy, numpy as np
import pandas, pandas as pd
import matplotlib.pyplot as plt

from miit.spatial_data.base_types import Annotation, Image, Pointset
from miit.spatial_data import Section


# def load_section(path):
#     histology_path = join(path, 'hist_image.png')
#     reference_image = Image.load_from_path(histology_path)
    
#     landmarks_path = join(path, 'landmarks.csv')
#     lms = Pointset.load_from_path(landmarks_path, name='landmarks')
    
#     tissue_mask_path = join(path, 'tissue_mask.png')
#     tissue_mask = Annotation.load_from_path(tissue_mask_path, name='tissue_mask')

#     section = Section(reference_image=reference_image, annotations=[lms, tissue_mask])
#     tissue_classes_path = join(path, 'tissue_classes.nii.gz')
#     labels_path = join(path, 'tissue_classes_labels.txt')
#     if exists(tissue_classes_path):
#         tissue_classes = Annotation.load_from_path(tissue_classes_path, path_to_labels=labels_path, name='tissue_classes')
#         section.annotations.append(tissue_classes)
#     return section

def load_section(directory):
    image_path = glob.glob(join(directory, 'images', '*'))[0]
    mask_path = glob.glob(join(directory, 'masks', '*'))[0]
    lm_path = glob.glob(join(directory, 'landmarks', '*'))[0]
    ann_path = glob.glob(join(directory, 'annotations', '*'))

    reference_image = Image.load_from_path(image_path)
    tissue_mask = Annotation.load_from_path(mask_path, name='tissue_mask')
    landmarks = Pointset.load_from_path(lm_path, name='landmarks')

    section = Section(reference_image=reference_image, annotations=[tissue_mask, landmarks])
    if len(ann_path) > 0:
        annotation_path = [x for x in ann_path if x.endswith('.nii.gz')][0]
        labels_path = [x for x in ann_path if x.endswith('.txt')][0]
        annotation = Annotation.load_from_path(annotation_path, path_to_labels=labels_path, name='tissue_classes')
        section.annotations.append(annotation)
    return section

def load_sections(root_dir):
    sections = OrderedDict()
    sub_dirs = sorted(os.listdir(root_dir), key=lambda x: int(x))
    for sub_dir in sub_dirs:
        sub_path = join(root_dir, sub_dir)
        section = load_section(sub_path)
        sections[sub_dir] = section
    return sections


def plot_sections(sections, with_landmarks: bool = True):
    """Plots histology, landmarks and tissue masks for each sections. 

    Args:
        sections (Section): Sections to plot. 
        with_landmarks (bool, optional): If True, plots landmarks in histology images. Defaults to True.
    """
    fig, axs = plt.subplots(len(sections), 2, figsize=(6 * 2, 6 * len(sections)))
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