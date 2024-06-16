import numpy as np
import pandas as pd
from scipy.integrate import trapezoid


def get_metabolite_intensities(msi, mz_intervals, spectra_idxs):
    norm_f = tic_trapz
    intensity_f = np.max
    gen_spec = False
    baseline_f = simple_baseline
    smooth_f = None

    intensities_per_spot = {}
    for spectrum_idx in spectra_idxs:
        if spectrum_idx not in intensities_per_spot:
            intensities_per_spot[spectrum_idx] = []
        collected_intensities = {}
        mzs, intensities = msi.getspectrum(spectrum_idx)
        # Smoothing is still missing
        intensities = baseline_f(intensities)
        for mz, radius, _ in mz_intervals:
            lower_bound = find_nearest(mzs, mz * (1 - radius))
            upper_bound = find_nearest(mzs, mz * (1 + radius))        
            intensity = norm_f(intensity_f(intensities[lower_bound[1]:upper_bound[1]]), intensities)        
            # intensities_per_spot[spectrum_idx].append((intensity, lower_bound, upper_bound))
            intensities_per_spot[spectrum_idx].append(intensity)
    return intensities_per_spot


def simple_baseline(intensities):
    return intensities - np.median(intensities[:100])


def tic_trapz(intensity, intensities, mz=None):
    return np.array(intensity) / trapezoid(y=intensities, x=mz)


def extract_msi_data(msi, mz_intervals, unique_ids):
    intensities_per_spot = get_metabolite_intensities(msi, mz_intervals, unique_ids)
    m_names = [x[2] for x in mz_intervals]
    intensities_df = pd.DataFrame(intensities_per_spot, index=m_names)
    return intensities_df.transpose()   


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def accumulate_counts_in_spot(source_keys, source_counts, intensity_df, background_identifier):
    bck_idx = np.argwhere(source_keys == background_identifier)
    bck_weight = source_counts[bck_idx].sum()
    source_keys = np.delete(source_keys, bck_idx)
    source_counts = np.delete(source_counts, bck_idx)
    selected_metabolites = intensity_df.loc[source_keys]    
    weights = source_counts/(source_counts.sum() + bck_weight)
    accumulated_metabolites = selected_metabolites.multiply(weights, axis=0).sum()
    return accumulated_metabolites


def accumulate_counts(mappings, intensity_df, valid_keys, background_idx):
    spot_wise_accumulated_data = []
    for target_key in mappings:
        source_keys, source_counts = mappings[target_key]
        accumulated_intensities = accumulate_counts_in_spot(source_keys, source_counts, intensity_df, background_idx)
        spot_wise_accumulated_data.append((accumulated_intensities, target_key))
    return pd.concat([col for (col, _) in spot_wise_accumulated_data], keys=[key for (_, key) in spot_wise_accumulated_data], axis=1) 
