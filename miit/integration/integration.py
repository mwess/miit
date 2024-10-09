from typing import  Dict, Set, Tuple, List, Optional

import numpy, numpy as np
import pandas, pandas as pd
from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.ImzMLParser import ImzMLParser

from miit.spatial_data.base_types.annotation import Annotation
from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics
from miit.spatial_data.spatial_omics.imzml import Imzml


def compute_reference_matrix_mappings(ref_mat1: numpy.ndarray, 
                                      ref_mat2: numpy.ndarray, 
                                      background1: int, 
                                      background2: int) -> Tuple[Dict[int, Tuple[numpy.ndarray, numpy.ndarray]], Dict[int, int]]:
    """Computes the composition of any pixel in ref_mat1 by ref_mat2. 

    Args:
        ref_mat1 (numpy.ndarray): 
        ref_mat2 (numpy.ndarray): 
        background1 (int): 
        background2 (int): 

    Returns:
        Tuple[Dict[int, Tuple[numpy.ndarray, numpy.ndarray]], Dict[int, int]]: 1. Mapping of each pixel in ref_mat1 to ref_mat2. 2. Mapping containing the amount of background pixel in each mapping.
    """
    px_composition = {}
    bck_masses = {}
    for i in range(ref_mat1.shape[0]):
        for j in range(ref_mat1.shape[1]):
            val = ref_mat1[i,j]
            if val == background1:
                continue
            if val not in bck_masses:
                bck_masses[val] = 0
            if val not in px_composition:
                px_composition[val] = []
            val2 = ref_mat2[i,j]
            if val2 == background2:
                bck_masses[val] += 1
            else:
                px_composition[val].append(val2)
    for key in px_composition:
        px_composition[key] = np.unique(px_composition[key], return_counts=True)
    return px_composition, bck_masses


def get_mappings(ref_mat1: numpy.ndarray, 
                 ref_mat2: numpy.ndarray, 
                 background1: int = 0, 
                 background2: int = 0) -> Tuple[Dict[int, numpy.ndarray], Dict[int, int], Set]:
    """
    Gets mappings for ref_mat1 from ref_mat2.
    Returns:
        mappings: mapping from ref_mat1 to ref_mat2
        spots_background: mass of background in each mapping.
        unique_vals: List of unique pixel values in ref_mat2 that are part of a mapping in ref_mat1.
    """
    mappings, spots_background = compute_reference_matrix_mappings(ref_mat1, ref_mat2, background1, background2)
    # Also get unique ids.
    unique_vals = set()
    for key in mappings:
        counts = mappings[key][0]
        for key2 in counts:
            if key2 != background2:
                unique_vals.add(key2)
    return mappings, spots_background, unique_vals


def accumulate_counts(mappings: Dict[int, Tuple[numpy.ndarray, numpy.ndarray]], 
                      measurement_df: pandas.core.frame.DataFrame, 
                      background_counts: Dict[int, int],
                      spot_accumulator_fun=None) -> pandas.core.frame.DataFrame:
    """Accumulated counts for each key using an accumulator_function.

    Args:
        mappings (Dict[int, Tuple[numpy.ndarray, numpy.ndarray]]): _description_
        measurement_df (pandas.core.frame.DataFrame): _description_
        background_counts (Dict[int, int]): _description_
        spot_accumulator_fun (_type_, optional): Function used to accumulate each single set of values. Defaults to None.

    Returns:
        _type_: _description_
    """
    spot_wise_accumulated_data = []
    for target_key in mappings:
        source_keys, source_counts = mappings[target_key]
        source_background = background_counts[target_key]
        accumulated_intensities = spot_accumulator_fun(source_keys, source_counts, measurement_df, source_background)
        spot_wise_accumulated_data.append((accumulated_intensities, target_key))
    final_df = pd.concat([col for (col, _) in spot_wise_accumulated_data], axis=0, ignore_index=True)
    final_df.rename(index={x[0]: x[1] for x in zip(range(len(spot_wise_accumulated_data)), [key for (_, key) in spot_wise_accumulated_data])}, inplace=True)
    return final_df


def get_number_of_background_pixels(df: pandas.core.frame.DataFrame, 
                                    background_value: int = -1) -> int:
    if background_value not in df.index:
        return 0
    return df.loc[background_value].shape[0]


def map_mapping_index_to_table_index(mapped_data: pandas.core.frame.DataFrame, 
                                     target_section: BaseSpatialOmics) -> pandas.core.frame.DataFrame:
    ref_to_spec_mapping = target_section.get_spec_to_ref_map(reverse=True)
    mapped_data = mapped_data.rename(index=ref_to_spec_mapping)
    return mapped_data


def transform_annotations_to_table(target_data: BaseSpatialOmics, 
                                   annotation: Annotation) -> pandas.core.frame.DataFrame:
    """Transforms annotations onto the spatial layout provided in spatial omics data.
    Note: At the moment only implemented for images in singlechannel mode (i.e. is_multichannel == False).

    Args:
        target_data (BaseSpatialOmics): Target spatial omics layout.
        annotation (Annotation): Source annotation.

    Returns:
        pandas.core.frame.DataFrame: Integrated annotations.
    """
    integrated_annotations = []
    annotation_data = annotation.data
    if len(annotation_data.shape) == 2:
        annotation_data = np.expand_dims(annotation_data, -1)
    if annotation.labels is not None:
        labels = annotation.labels
    else:
        # Just use indices
        n_annotations = annotation_data.shape[2]
        labels = list(range(n_annotations))
    integrated_annotations = map_annotations_to_table(target_data.get_spec_to_ref_map(), 
                                                      target_data.ref_mat.data,
                                                      annotation_data, 
                                                      labels)
    return integrated_annotations


def map_annotations_to_table(spec_to_ref_map: Dict, 
                             ref_mat: numpy.ndarray, 
                             annotations: numpy.ndarray, 
                             labels: List[str]) -> pandas.core.frame.DataFrame: 
    glob_counts = {spec_to_ref_map[x]: np.zeros(annotations.shape[2]) for x in spec_to_ref_map}
    spot_counts = {spec_to_ref_map[x]: 0 for x in spec_to_ref_map}
    for i in range(ref_mat.shape[0]):
        for j in range(ref_mat.shape[1]):
            val = ref_mat[i,j]
            if val == 0:
                continue
            glob_counts[val] += annotations[i, j, :]
            spot_counts[val] += 1
    for key in glob_counts:
         if spot_counts[key] > 0:
             glob_counts[key] /= spot_counts[key]
    count_df = pd.DataFrame(glob_counts)
    count_df.rename(columns={spec_to_ref_map[x]: x for x in spec_to_ref_map}, 
                    index={idx: name for (idx, name) in zip(range(len(labels)), labels)}, 
                    inplace=True)
    count_df = count_df.transpose()
    return count_df

    
def map_accumulated_data_to_imzml(target_bmi: Imzml,
                                  accumulated_df: pandas.core.frame.DataFrame,
                                  output_path: str,
                                  mzs: Optional[numpy.ndarray] = None):
    spec_to_ref_map = target_bmi.get_spec_to_ref_map()
    template_imzml = ImzMLParser(target_bmi.config['imzml'])
    if mzs is None:
        mzs = np.array([float(x) for x in accumulated_df.columns])
    imzml_writer = ImzMLWriter(output_path)
    spec_to_ref_map = target_bmi.get_spec_to_ref_map()
    for msi_idx, coords in enumerate(template_imzml.coordinates):
        acc_df_idx = spec_to_ref_map[msi_idx]
        intensities = accumulated_df.loc[acc_df_idx].to_numpy()
        imzml_writer.addSpectrum(mzs, intensities, coords)
    imzml_writer.finish()
    
    