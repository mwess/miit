from typing import Dict, List, Tuple

import numpy, numpy as np
import pandas, pandas as pd
import pyimzml
from scipy.integrate import trapezoid


def get_metabolite_intensities(msi: pyimzml.ImzMLParser.ImzMLParser, 
                               mz_intervals: Tuple[float, float, float], 
                               spectra_idxs: List[int]) -> Dict[int, numpy.ndarray]:
    norm_f = tic_trapz
    intensity_f = np.max
    baseline_f = simple_baseline

    intensities_per_spot = {}
    for spectrum_idx in spectra_idxs:
        if spectrum_idx not in intensities_per_spot:
            intensities_per_spot[spectrum_idx] = []
        mzs, intensities = msi.getspectrum(spectrum_idx)
        # Smoothing is still missing
        intensities = baseline_f(intensities)
        for mz, radius, _ in mz_intervals:
            lower_bound = find_nearest(mzs, mz * (1 - radius))
            upper_bound = find_nearest(mzs, mz * (1 + radius))        
            intensity = norm_f(intensity_f(intensities[lower_bound[1]:upper_bound[1]]), intensities)        
            intensities_per_spot[spectrum_idx].append(intensity)
    return intensities_per_spot


def simple_baseline(intensities: numpy.ndarray) -> numpy.ndarray:
    return intensities - np.median(intensities[:100])


def tic_trapz(intensity: float, intensities: numpy.ndarray, mz=None) -> numpy.ndarray:
    return np.array(intensity) / trapezoid(y=intensities, x=mz)


def extract_msi_data(msi: pyimzml.ImzMLParser.ImzMLParser, 
                     mz_intervals: Tuple[float, float, float], 
                     unique_ids: List[int]) -> pandas.core.frame.DataFrame:
    intensities_per_spot = get_metabolite_intensities(msi, mz_intervals, unique_ids)
    m_names = [x[2] for x in mz_intervals]
    intensities_df = pd.DataFrame(intensities_per_spot, index=m_names)
    return intensities_df.transpose()   


def find_nearest(array: numpy.ndarray, value: float) -> Tuple[float, int]:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def accumulate_counts_in_spot(source_keys: numpy.ndarray, 
                              source_counts: numpy.ndarray, 
                              intensity_df: pandas.core.frame.DataFrame, 
                              background_identifier: int) -> pandas.core.frame.DataFrame:
    bck_idx = np.argwhere(source_keys == background_identifier)
    bck_weight = source_counts[bck_idx].sum()
    source_keys = np.delete(source_keys, bck_idx)
    source_counts = np.delete(source_counts, bck_idx)
    selected_metabolites = intensity_df.loc[source_keys]    
    weights = source_counts/(source_counts.sum() + bck_weight)
    accumulated_metabolites = selected_metabolites.multiply(weights, axis=0).sum()
    return accumulated_metabolites


def accumulate_counts(mappings: Dict[int, Tuple[numpy.ndarray, numpy.ndarray]], 
                      intensity_df: pandas.core.frame.DataFrame, 
                      background_idx: int) -> pandas.core.frame.DataFrame:
    spot_wise_accumulated_data = []
    for target_key in mappings:
        source_keys, source_counts = mappings[target_key]
        accumulated_intensities = accumulate_counts_in_spot(source_keys, source_counts, intensity_df, background_idx)
        spot_wise_accumulated_data.append((accumulated_intensities, target_key))
    return pd.concat([col for (col, _) in spot_wise_accumulated_data], keys=[key for (_, key) in spot_wise_accumulated_data], axis=1) 
