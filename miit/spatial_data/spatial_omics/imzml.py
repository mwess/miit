from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from os.path import join, exists
from pathlib import Path
from typing import (
    Any, 
    Callable,
    ClassVar, 
    Optional, 
    Dict
)
import uuid

from lxml import etree
import pandas, pandas as pd
import numpy, numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
import pyimzml
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter

from miit.spatial_data.base_types import (
    Annotation,
    BaseImage,
    Image,
    BasePointset,
    SpatialBaseDataLoader
)
from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics
from miit.registerers.base_registerer import Registerer
from miit.utils.imzml import (
    get_mode,
    get_scan_direction,
    get_scan_pattern,
    get_scan_type,
    get_line_scan_direction,
    get_pca_img
)


def compose_dicts(dict1: dict, dict2: dict) -> dict:
    """Composes two dictionaries. If a key is only present in one of the two dictionaries, it
    will be skipped.

    Args:
        dict1 (dict): 
        dict2 (dict):

    Returns:
        dict: Composed dictionaries.
    
    """
    return {k: dict2.get(v) for k, v in dict1.items() if v in dict2}

def export_imzml(template_msi: pyimzml.ImzMLParser.ImzMLParser, 
                 output_path: str,
                 integrated_data: pandas.DataFrame) -> None:
    """Exports integrated msi data into the imzML format. Most of the work is done by
    `pyimzml`'s `ImzMLWriter`. However some information such as pixel size is not provided, which
    we add by manually writing the information to the imzML file.

    
    Args:
        template_msi (pyimzml.ImzMLParser.ImzMLParser): Template msi. Defines target topology.
        output_path (str): 
        integrated_data (pandas.DataFrame): DataFrame containing integrated msi data.
    """

    mzs = integrated_data.columns.to_numpy()
    if mzs.dtype != np.float64:
        mzs = mzs.astype(np.float64)
    mode = get_mode(template_msi)
    scan_direction = get_scan_direction(template_msi)
    line_scan_direction = get_line_scan_direction(template_msi)
    scan_pattern = get_scan_pattern(template_msi)
    scan_type = get_scan_type(template_msi)

    if mzs.dtype != np.float64:
        mzs = mzs.astype(np.float64)

    with ImzMLWriter(output_path, 
                     mz_dtype=np.float64, 
                     intensity_dtype=np.float32, 
                     mode=mode,
                     scan_direction=scan_direction,
                     line_scan_direction=line_scan_direction,
                     scan_pattern=scan_pattern,
                     scan_type=scan_type,
                     polarity='negative') as writer:
        for i, (x, y, z) in enumerate(template_msi.coordinates):
            if i not in integrated_data.index:
                continue
            intensities = integrated_data.loc[i].to_numpy()
            if intensities.dtype != np.float32:
                intensities = intensities.astype(np.float32)
            writer.addSpectrum(mzs, intensities, (x, y, z))
    # Now we add additional parameters that imzml skipped.
    scan_settings_params = [
        ("max dimension x", "IMS:1000044", template_msi.imzmldict['max dimension x']),
        ("max dimension y", "IMS:1000045", template_msi.imzmldict['max dimension y']),
        ("pixel size x", "IMS:1000046", template_msi.imzmldict['pixel size x']),
        ("pixel size y", "IMS:1000047", template_msi.imzmldict['pixel size y']),
    ]
    # scan_settings_params = []
    sl = "{http://psi.hupo.org/ms/mzml}"
    elem_iterator = etree.parse(output_path)
    root = elem_iterator.getroot()
    scan_settings_list_elem = root.find('%sscanSettingsList' % sl)
    first_scan_setting = scan_settings_list_elem.find('./%sscanSettings' %sl)
    first_cv_param_elem = first_scan_setting.findall('./')[0]
    for (name, accession, value) in scan_settings_params:
        template_cv_param_elem = deepcopy(first_cv_param_elem)
        template_cv_param_elem.attrib['accession'] = accession
        template_cv_param_elem.attrib['value'] = str(value)
        template_cv_param_elem.attrib['name'] = name
        first_scan_setting.append(template_cv_param_elem)
    xml_as_str = etree.tostring(root, pretty_print=True)
    with open(output_path, 'wb') as f:
        f.write(xml_as_str)


def simple_baseline(intensities: numpy.ndarray) -> numpy.ndarray:
    """Computes a baseline for intensities by subtracting the median intensity of the first 100 intensities.

    Args:
        intensities (numpy.ndarray):

    Returns:
        numpy.ndarray:
    """
    return intensities - np.median(intensities[:100])


def find_nearest(array: numpy.ndarray, value: float) -> tuple[float, int]:
    """Find the index and value in the array closest to value.

    Args:
        array (numpy.ndarray): 
        value (float): 

    Returns:
        tuple[float, int]: Closest value, index of closest value.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def tic_trapz(intensity: float, 
              intensities: numpy.ndarray, 
              mz: float | None = None) -> numpy.ndarray:
    """
    Normalizes intensity by using the integral of the intensities.
    Args:
        intensity (float):
        intensities (numpy.ndarray):
        mz (float | None, optional):Defaults to None.

    Returns:
        numpy.ndarray:
    """
    return np.array(intensity) / trapezoid(y=intensities, x=mz)


def get_metabolite_intensities(
        msi: pyimzml.ImzMLParser.ImzMLParser, 
        mz_dict: Dict[int | str, tuple[float, float]], 
        spectra_idxs: set[int],
        norm_f: Callable | None = None,
        intensity_f: Callable | None = None,
        baseline_f: Callable | None = None) -> Dict[int | str, list[float]]:
    """Computes metabolite intensities from the given msi data.

    Args:
        msi (pyimzml.ImzMLParser.ImzMLParser): Souce msi.
        mz_dict (Dict[Any, tuple[float, float]]): Dict mapping to mz and radius.
        spectra_idxs (set[int]): List of spectra to extract data from. 
        norm_f (Callable | None, optional): Normalisation function. If None, uses `tic_trapz`. Defaults to None.
        intensity_f (Callable | None, optional): Intensity function, If None, takes the maximum value of the intensity. Defaults to None.
        baseline_f (Callable | None, optional): Baseline function. If None, uses `simple_baseline`. Defaults to None.

    Returns:
        dict[int | str, list[float]]: Dict mapping from to computed intensity values.
    """
    if norm_f is None:
        norm_f = tic_trapz
    if intensity_f is None:
        intensity_f = np.max
    if baseline_f is None:
        baseline_f = simple_baseline
    
    intensities_per_spectrum = {}
    for spectrum_idx in spectra_idxs:
        if spectrum_idx not in intensities_per_spectrum:
            intensities_per_spectrum[spectrum_idx] = []
        mzs, intensities = msi.getspectrum(spectrum_idx)
        # Smoothing is still missing
        intensities = baseline_f(intensities)
        for key in mz_dict:
            mz_elem = mz_dict[key]
            mz, radius = mz_elem['interval']
            lower_bound = find_nearest(mzs, mz * (1 - radius))
            upper_bound = find_nearest(mzs, mz * (1 + radius))                    
            intensity = norm_f(intensity_f(intensities[lower_bound[1]:upper_bound[1]]), intensities)        
            intensities_per_spectrum[spectrum_idx].append(intensity)
    return intensities_per_spectrum


def get_metabolite_intensities_from_full_spectrum(msi: pyimzml.ImzMLParser.ImzMLParser,
                                                  spectra_idxs: list[int], 
                                                  mz_intervals: tuple[float, float, float],
                                                  norm_f: Callable[[numpy.ndarray], numpy.ndarray] | None = None,
                                                  baseline_f: Callable[[numpy.ndarray], numpy.ndarray] | None = None) -> dict:
    """Identifies intensity peaks based on the list of provided `mz_intervals` in `msi`. `baseline_f` can be
    used to preprocess intensities, `norm_f` is used to determine the intensity value within the given mz_interval. 
    Only spectra within `spectra_idxs` will be processed.

    Args:
        msi (pyimzml.ImzMLParser.ImzMLParser): _description_
        spectra_idxs (list[int]): _description_
        mz_intervals (tuple[float, float, float]): _description_
        norm_f (Callable[[numpy.ndarray], numpy.ndarray] | None, optional): _description_. Defaults to None.
        baseline_f (Callable[[numpy.ndarray], numpy.ndarray] | None, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """
    if norm_f is None:
        norm_f = tic_trapz
    if intensity_f is None:
        intensity_f = np.max
    if baseline_f is None:
        baseline_f = simple_baseline
    
    intensities_per_spot = {}
    for spectrum_idx in spectra_idxs:
        if spectrum_idx not in intensities_per_spot:
            intensities_per_spot[spectrum_idx] = []
        mzs, intensities = msi.getspectrum(spectrum_idx)
        intensities = baseline_f(intensities)
        for start, end, _ in mz_intervals:
            lower_bound = find_nearest(mzs, start)
            upper_bound = find_nearest(mzs, end)        
            intensity = norm_f(intensity_f(intensities[lower_bound[1]:upper_bound[1]]), intensities)        
            intensities_per_spot[spectrum_idx].append(intensity)
    return intensities_per_spot


# TODO: Functions for getting metabolites can be merged together.
def get_metabolite_intensities_preprocessed(msi: pyimzml.ImzMLParser.ImzMLParser,
                                            spectra_idxs: set[int],
                                            mz_intervals: list[dict] | None = None) -> dict[int | str, list[float]]:
    """Extracts metabolites from imzml file. Assumes that targets have been preprocessed and selected in SCiLS prior to exporting.

    Args:
        msi (pyimzml.ImzMLParser.ImzMLParser): 
        spectra_idxs (set[int]): 
        mz_intervals (list[dict] | None, optional): . Defaults to None.

    Returns:
        dict[int | str, list[float]]: _description_
    """
    intensity_f = np.max
    intensities = {}
    for spectrum_idx in spectra_idxs:
        if spectrum_idx not in intensities:
            intensities[spectrum_idx] = []
        mzs, intensities = msi.getspectrum(spectrum_idx)
        for idx, key in enumerate(mz_intervals):
            mz_elem = mz_intervals[key]
            mz, radius = mz_elem['interval']
            intensity = intensities[idx]
            lower_bound = find_nearest(mzs, mz - radius)
            upper_bound = find_nearest(mzs, mz + radius)
            lower = lower_bound[1]
            upper = upper_bound[1]
            if lower == upper:
                upper = upper + 1
            intensity = intensity_f(intensities[lower:upper])
            intensities[spectrum_idx].append(intensity)
    return intensities


def get_metabolite_intensities_targeted(msi: pyimzml.ImzMLParser.ImzMLParser,
                                        spectra_idxs: set[int],
                                        mz_labels=None) -> pd.DataFrame:
#tuple[dict[int | str, list[float]], list[str]]:
    """Simply assume that metabolites are already preprocessed and we just want to collect them.

    Args:
        msi (pyimzml.ImzMLParser.ImzMLParser): 
        spectra_idxs (set[int]): _description_
        mz_labels (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple[dict[int | str, list[float]], list[str]]: _description_
    """
    collected_intensities = {}
    for spectrum_idx in spectra_idxs:
        mzs, intensities = msi.getspectrum(spectrum_idx)
        collected_intensities[spectrum_idx] = intensities.copy()
    if mz_labels is None:
        mz_labels = ["{:10.3f}".format(x).strip() for x in mzs]
    metabolite_df = pd.DataFrame(collected_intensities, index=mz_labels)
    return metabolite_df


def convert_msi_to_reference_matrix(msi: pyimzml.ImzMLParser.ImzMLParser, 
                                    target_resolution: int = 1) -> numpy.ndarray | dict:
    """Computes reference matrix from msi scaled to target_resolution.

    Args:
        msi (pyimzml.ImzMLParser.ImzMLParser): Source imzml.
        target_resolution (int, optional): Target resolution to which. Defaults to 1.


    Returns:
        numpy.ndarray | dict: Reference matrix, mapping of msi pixels to reference matrix.
    """
    scale_x = msi.imzmldict['pixel size x']/target_resolution
    scale_y = msi.imzmldict['pixel size y']/target_resolution
    max_x = int(msi.imzmldict['max dimension x']/target_resolution)
    max_y = int(msi.imzmldict['max dimension y']/target_resolution)
    ref_mat = np.zeros((max_y, max_x), dtype=int)
    spec_to_ref_map = {}
    ref_map_to_spec = {}
    for idx, (x,y,_) in enumerate(msi.coordinates):
        x_s = int((x-1)*scale_x)
        x_e = int(x_s + scale_x)
        y_s = int((y-1)*scale_y)
        y_e = int(y_s + scale_y)
        ref_mat[y_s:y_e,x_s:x_e] = idx + 1
        spec_to_ref_map[idx] = idx + 1
        ref_map_to_spec[idx + 1] = idx
    ann = Annotation(data=ref_mat, labels=ref_map_to_spec)
    return ann, spec_to_ref_map


# TODO: Remove that function.
def convert_to_matrix(msi: pyimzml.ImzMLParser.ImzMLParser, 
                      srd: dict = None, 
                      target_resolution: int = 1) -> numpy.ndarray | dict | Optional[numpy.ndarray]:
    """Computes reference matrix from msi scaled to target_resolution. Will also convert a srd annotation to
    a binary annotation matrix, if provided. 

    Args:
        msi (pyimzml.ImzMLParser.ImzMLParser): Source imzml.
        srd (dict, optional): Additional srd annotation (from SCiLS). Defaults to None.
        target_resolution (int, optional): Target resolution to which. Defaults to 1.


    Returns:
        Union[numpy.ndarray, dict, Optional[numpy.ndarray]]: Reference matrix, mapping of msi pixels to reference matrix, If supplied, srd in matrix form.
    """
    scale_x = msi.imzmldict['pixel size x']/target_resolution
    scale_y = msi.imzmldict['pixel size y']/target_resolution
    max_x = int(msi.imzmldict['max dimension x']/target_resolution)
    max_y = int(msi.imzmldict['max dimension y']/target_resolution)
    proj_mat = np.zeros((max_y, max_x), dtype=int)
    spec_to_ref_map = {}
    for idx, (x,y,_) in enumerate(msi.coordinates):
        x_s = int((x-1)*scale_x)
        x_e = int(x_s + scale_x)
        y_s = int((y-1)*scale_y)
        y_e = int(y_s + scale_y)
        proj_mat[y_s:y_e,x_s:x_e] = idx + 1
        spec_to_ref_map[idx] = idx + 1
    if srd is not None:
        annotation_mat = np.zeros((max_y, max_x), dtype=np.uint8)
        points = []
        for _, point in enumerate(srd['Regions'][0]['Sources'][0]['Spots']):
            x = point['X']
            y = point['Y']
            points.append((x,y))
        points = np.array(points)
        points[:, 0] = points[:,0] - np.min(points[:,0])
        points[:, 1] = points[:,1] - np.min(points[:,1])
        for i in range(points.shape[0]):
            x = points[i,0]
            y = points[i,1]
            x_s = int(x*scale_x)
            x_e = int(x_s + scale_x)
            y_s = int(y*scale_y)
            y_e = int(y_s + scale_y)
            annotation_mat[y_s:y_e,x_s:x_e] = 1
    else:
        annotation_mat = None
    return proj_mat, spec_to_ref_map, annotation_mat


def compute_mean_spectrum(msi: pyimzml.ImzMLParser.ImzMLParser) -> numpy.ndarray:
    """Compute the mean spectrum for all spectra.

    Args:
        msi (pyimzml.ImzMLParser.ImzMLParser):

    Returns:
        numpy.ndarray: Mean spectra.
    """
    total_intensities = None
    for i in range(len(msi.coordinates)):
        mzs, intensities = msi.getspectrum(i)
        if total_intensities is None:
            total_intensities = intensities.copy()
        else:
            total_intensities += intensities
    avg_spec = total_intensities/len(msi.coordinates)
    return avg_spec


# TODO: Remove. This is just custom code.
def load_metabolites(table_path: str, 
                     imzml_path: str) -> tuple[dict, pandas.DataFrame]:
    # NEDC_peak_table = pd.read_csv('Peaklist_136_NEDC_figshare.txt', sep='\t')
    NEDC_peak_table = pd.read_csv(table_path, sep='\t')
    NEDC_peak_table_IDed = NEDC_peak_table[NEDC_peak_table['ID'].notna()][['m/z', 'ID', 'ID in OPLSDA']].reset_index(drop=True)
    msi = ImzMLParser(imzml_path)
    peaks = get_peaks(msi, rel_percentage=0.00005)
    peak_dict, peak_intervals = get_one_peak_dict_and_interval_list(find_ided_peaks(peaks, NEDC_peak_table_IDed))
    return peak_dict, NEDC_peak_table_IDed


# TODO: Remove function.
def get_peaks(msi: pyimzml.ImzMLParser.ImzMLParser, rel_percentage=0.00025):
    mzs = msi.getspectrum(0)[0]
    mean_intensities = compute_mean_spectrum(msi)
    norm_intensities = mean_intensities / trapezoid(y=mean_intensities, x=None)
    norm_intensities = 100 * norm_intensities /  norm_intensities.max()
    peaks, _ = find_peaks(norm_intensities, height=rel_percentage)
    p_m = [(mean_intensities, mzs)]
    return p_m


# TODO: Remove function.
def find_ided_peaks(peaks, peak_table, mass_error_mz=2.00000):
    peak_max_diffs = np.abs(peak_table['m/z'] - peak_table['m/z'] * (1 + mass_error_mz))
    _ret = []
    for _idxs, _mzs in peaks:
        _t_idx = []
        _t_mz = []
        _t_id = []
        for _idx, _mz in zip(_idxs, _mzs):
            peak_table_val = peak_table['m/z']
            diffs = np.abs(peak_table_val - _mz)
            sub_table = peak_table[diffs < peak_max_diffs]
            if sub_table.shape[0] > 0:
                _t_idx.append(_idx)
                _t_mz.append(_mz)
                _t_id.append(peak_table['ID'].iloc[(np.abs(peak_table['m/z'] - _mz)).argmin()])
        _ret.append((_t_idx, _t_mz, _t_id))
    return _ret


# TODO: Remove
def get_one_peak_dict_and_interval_list(peaks_id: tuple[int, float, float], 
                                        delta_factor: int = 2,
                                        default_interval_delta: float = 0.00025):
    _ret = {}
    _ret_ints = []
    for _idxs, _mzs, _ids in peaks_id:
        for _idx, _mz, _id in zip(_idxs, _mzs, _ids):
            if not _id in _ret:
                _ret[_id] = {}
                _ret[_id]['mzs'] = []
                _ret[_id]['interval'] = None
            _ret[_id]['mzs'].append(_mz)
    for _id, _data in _ret.items():
        _d = np.max(_ret[_id]['mzs']) - np.min(_ret[_id]['mzs'])
        if _d > 0:
            _int_delta = delta_factor * _d / np.mean(_ret[_id]['mzs'])
        else:
            _int_delta = default_interval_delta
        _ret[_id]['interval'] = (np.mean(_ret[_id]['mzs']), _int_delta)
        _ret_ints.append(_ret[_id]['interval'])
    return _ret, _ret_ints


def compute_weighted_average(measurements: pandas.DataFrame | numpy.ndarray,
                             weights: numpy.ndarray, 
                             background_weight: float) -> pandas.DataFrame:
    """Compute weighted average for measurements. Weight is computed as the sum of weights + background_weight.

    Args:
        measurements (pandas.DataFrame | numpy.ndarray):
        weights (numpy.ndarray): 
        background_weight (float):

    Returns:
        pandas.DataFrame:
    """
    return (measurements*weights).sum()/(sum(weights) + background_weight)


def msi_default_accumulate_spot_weighted_mean(source_keys: numpy.ndarray,
                                              source_counts: numpy.ndarray,
                                              measurement_df: pandas.DataFrame,
                                              bck_weight: float) -> pandas.DataFrame:
    """Compute the weighted mean for each spot defined in source_keys.

    Args:
        source_keys (numpy.ndarray): 
        source_counts (numpy.ndarray): 
        measurement_df (pandas.DataFrame): 
        bck_weight (float): 

    Returns:
        pandas.DataFrame: 
    """
    selected_datas = measurement_df[source_keys].transpose()
    return pd.DataFrame(selected_datas.apply(lambda x: compute_weighted_average(x, source_counts, bck_weight), axis=0)).transpose()
  

def msi_default_spot_accumulation_fun(source_keys: numpy.ndarray, 
                                      source_counts: numpy.ndarray, 
                                      measurement_df: pandas.DataFrame, 
                                      bck_weight: float,
                                      accumulator_function: Callable | None = None) -> pandas.DataFrame:
    """
    Default function for accumulating spots. If not specific accumulator function is supplied, computes the following 
    statistics for each accumulated spots: mean, std, min, max, median.
    
    Args:
        source_keys (numpy.ndarray): 
        source_counts (numpy.ndarray): 
        measurement_df (pandas.DataFrame): 
        bck_weight (float): 
        accumulator_function (Callable | None, optional): If None, computes mean, std, min, max, median. Defaults to None.

    Returns:
        pandas.DataFrame: 
    """
    if accumulator_function is None:
        accumulator_function = lambda r: pd.Series({'mean': r.mean(), 
                                                    'std': r.std(), 
                                                    'min': r.min(), 
                                                    'max': r.max(), 
                                                    'median': r.median()})
    unrolled_keys = np.repeat(source_keys, source_counts)
    selected_datas = measurement_df[source_keys].transpose() 
    if unrolled_keys.shape[0] == 0:
        unrolled_datas_stats = selected_datas.copy()
        unrolled_datas = pd.DataFrame(np.zeros((bck_weight, selected_datas.shape[1])), 
                                      index=bck_weight*['background'], 
                                      columns=selected_datas.columns)
    else:
        unrolled_datas = selected_datas.loc[unrolled_keys]
        if bck_weight > 0:
            zero_df = pd.DataFrame(np.zeros((bck_weight, unrolled_datas.shape[1])), 
                                             columns=unrolled_datas.columns,
                                             index=bck_weight*['background'])
            unrolled_datas = pd.concat([unrolled_datas, zero_df], axis=0)
    accumulated_vals = unrolled_datas.apply(accumulator_function).transpose()
    unrolled_datas_stats = flatten_to_row(accumulated_vals)
    unrolled_datas_stats['n_bck_pixls'] = bck_weight
    # Background pixel information here
    unrolled_datas_stats['n_pixels'] = unrolled_keys.shape[0] + bck_weight 
    return unrolled_datas_stats


def flatten_to_row(df: pandas.DataFrame) -> pandas.DataFrame:
    """Flattens a dataframe to a row.

    Args:
        df (pandas.DataFrame): 

    Returns:
        pandas.DataFrame: 
    """
    v = df.unstack().to_frame().sort_index(level=1).T
    v.columns = v.columns.map('_'.join)    
    return v


# TODO: Use ref_mat labels instead of spec_to_ref_map, but everything needs to be reverted then.
@dataclass
class Imzml(BaseSpatialOmics):

    __ref_mat: Annotation = field(init=False, default=None)
    spec_to_ref_map: dict
    additional_spatial_data: list[BaseImage | BasePointset] = field(default_factory=lambda: [])
    background: ClassVar[int] = 0
    config: dict | None = None
    name: str = ''
    msi: pyimzml.ImzMLParser.ImzMLParser | None = None

    def __post_init__(self):
        self._id = uuid.uuid1()
        self.msi = ImzMLParser(self.config['imzml'])

    @property
    def ref_mat(self):
        return self.__ref_mat
    
    @ref_mat.setter
    def ref_mat(self, ref_mat: Annotation):
        self.__ref_mat = ref_mat

    @staticmethod
    def get_type() -> str:
        return 'Imzml'

    def pad(self, padding: tuple[int, int, int, int]):
        self.__ref_mat.pad(padding, constant_values=self.background)
        for spatial_data in self.additional_spatial_data:
            spatial_data.pad(padding)

    def resize(self, height: int, width: int):
        w, h = self.__ref_mat.data[:2]
        ws = w // width
        hs = h // height        
        self.__ref_mat.resize(height, width)
        for spatial_data in self.additional_spatial_data:
            if isinstance(spatial_data, BasePointset):
                spatial_data.resize(ws, hs)
            else:
                spatial_data.resize(width, height)

    def rescale(self, scaling_factor: float):
        self.__ref_mat.rescale(scaling_factor)
        for spatial_data in self.additional_spatial_data:
            spatial_data.rescale(scaling_factor)

    def crop(self, x1: int, x2: int, y1: int, y2: int):
        self.__ref_mat.crop(x1, x2, y1, y2)
        for spatial_data in self.additional_spatial_data:
            spatial_data.crop(x1, x2, y1, y2)

    def get_spec_to_ref_map(self, reverse: bool = False):
        map_ = None
        if reverse:
            map_ = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        else:
            map_ = self.spec_to_ref_map.copy()
        return map_

    def copy(self):
        ref_mat = self.__ref_mat.copy()
        spec_to_ref_map = self.spec_to_ref_map.copy()
        obj = Imzml(
            config=self.config.copy(),
            spec_to_ref_map=spec_to_ref_map,
            additional_spatial_data=self.additional_spatial_data.copy()
        )
        obj.ref_mat = ref_mat
        return obj

    def apply_transform(self, 
             registerer: Registerer, 
             transformation: Any, 
             **kwargs: dict) -> 'Imzml':
        ref_mat_transformed = self.__ref_mat.apply_transform(registerer, transformation, **kwargs)
        transformed_spatial_datas = []
        for spatial_data in self.additional_spatial_data:
            transformed_spatial_data = spatial_data.apply_transform(registerer, transformation, **kwargs)
            transformed_spatial_datas.append(transformed_spatial_data)
        config = self.config.copy() if self.config is not None else None
        scils_export_imzml_transformed = Imzml(
            config=config,
            spec_to_ref_map=self.spec_to_ref_map,
            additional_spatial_data=transformed_spatial_datas,
            name=self.name)
        scils_export_imzml_transformed.ref_mat = ref_mat_transformed
        return scils_export_imzml_transformed

    def flip(self, axis: int = 0):
        w, h = self.__ref_mat.shape[:2]
        self.__ref_mat.flip(axis=axis)
        for spatial_data in self.additional_spatial_data:
            if isinstance(spatial_data, BasePointset):
                spatial_data.flip((w, h), axis=axis)
            else:
                spatial_data.flip(axis=axis)

    def store(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        f_dict = {}
        config_path = join(directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
        f_dict['config_path'] = config_path
        self.__ref_mat.store(join(directory, 'ref_mat'))
        f_dict['__ref_mat'] = join(directory, 'ref_mat')
        spec_to_ref_map_path = join(directory, 'spec_to_ref_map.json')
        with open(spec_to_ref_map_path, 'w') as f:
            json.dump(self.spec_to_ref_map, f)
        f_dict['spec_to_ref_map_path'] = spec_to_ref_map_path
        sd_ids = []
        for spatial_data in self.additional_spatial_data:
            spatial_data.store(join(directory, str(spatial_data._id)))
            sd_ids.append(
                {
                    'id': str(spatial_data._id),
                    'type': spatial_data.get_type()
                }
            )
        f_dict['additional_spatial_data'] = sd_ids
        f_dict['id'] = str(self._id)

        f_dict['name'] = self.name
        with open(join(directory, 'attributes.json'), 'w') as f:
            json.dump(f_dict, f)

    @classmethod
    def load(cls, 
             directory: str,
             spatial_base_data_loader: SpatialBaseDataLoader | None = None) -> 'Imzml':
        if spatial_base_data_loader is None:
            spatial_base_data_loader = SpatialBaseDataLoader.load_default_loader()
        with open(join(directory, 'attributes.json')) as f:
            attributes = json.load(f)
        with open(attributes['config_path']) as f:
            config = json.load(f)
        __ref_mat = Annotation.load(attributes['__ref_mat'])
        with open(attributes['spec_to_ref_map_path']) as f:
            spec_to_ref_map = json.load(f)
            # Clean dictionary values to int
            spec_to_ref_map = {int(x): int(spec_to_ref_map[x]) for x in spec_to_ref_map}
        additional_spatial_data = []
        for spatial_data_dict in attributes['additional_spatial_data']:
            sub_dir = spatial_data_dict['id']
            spatial_data = spatial_base_data_loader.load(spatial_data_dict['type'], join(directory, sub_dir))
            additional_spatial_data.append(spatial_data)            
        name = attributes.get('name', '')
        obj = cls(
            config=config,
            additional_spatial_data=additional_spatial_data,
            spec_to_ref_map=spec_to_ref_map,
            name=name
        ) 
        obj.ref_mat = __ref_mat
        id_ = uuid.UUID(attributes['id'])
        obj._id = id_
        return obj
        
    @classmethod
    def init_msi_data(cls,
                      imzml_path: str,
                      config: dict | None = None,
                      target_resolution: int | None = 1,
                      name: str = '') -> 'Imzml':
        """Inits an Imzml object. 

        Args:
            imzml_path (str): Path to imzml path.
            config (dict | None, optional): Contains additional user defined data. Defaults to None.
            target_resolution (int | None, optional): Resolution to which to rescale the msi data. Defaults to 1.
            name (str, optional): Name of the imzml object. Defaults to ''.

        Returns:
            Imzml:
        """
        if config is None:
            config = {}
        if 'imzml' not in config:
            config['imzml'] = imzml_path
        msi = ImzMLParser(imzml_path)
        ref_mat, spec_to_ref_map = convert_msi_to_reference_matrix(msi, target_resolution)
        obj = cls(
            config=config,
            spec_to_ref_map=spec_to_ref_map,
            name=name
        )
        obj.ref_mat = ref_mat
        return obj
            
    # TODO: Can this be removed
    def convert_mappings_and_unique_ids_back(self, 
                                             mappings: dict, 
                                             unique_ids: set) -> tuple[dict, set]:
        """Helper function to convert internal and external ids. 

        Args:
            mappings (dict): 
            unique_ids (set): 

        Returns:
            tuple[dict, set]: 
        """
        for key in mappings:
            mappings[key] = mappings[key] - 1
        unique_ids = {x - 1 for x in unique_ids}
        return mappings, unique_ids

    def get_map_to_msi_pixel_idxs(self, ref_mat_values: set | None = None) -> set:
        """Retrieves spectra ids.

        Args:
            ref_mat_values (set | None, optional): Reference matrix ids to use as an additional filter. Defaults to None.

        Returns:
            set:
        """
        # Invert map
        ref_to_spec_map = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        if ref_mat_values is None:
            ref_mat_values = ref_to_spec_map.keys()
        return {int(ref_to_spec_map[x]) for x in ref_mat_values}

    # TODO: Is this needed?
    def mappings_map_to_msi_pixel_idxs(self, mappings: dict) -> dict:
        ref_to_spec_map = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        mapped_mappings = {}
        for key in mappings:
            idx_arr = mappings[key][0]
            idx_arr_mapped = np.array([int(ref_to_spec_map[x]) for x in idx_arr])
            mapped_mappings[key] = (idx_arr_mapped, mappings[key][1].copy())
        return mapped_mappings
    
    def get_pca_img(self,
                    int_threshold: float | None = None) -> Image:
        """Compute the PCA image representation of the msi data.

        Args:
            int_threshold (Optional[float], optional): Intensity threshold. Defaults to None.

        Returns:
            Image: PCA image presentation
        """
        return Image(data=get_pca_img(
            self.msi,
            self.ref_mat.data,
            self.spec_to_ref_map,
            int_threshold
        ))

    
    def extract_ion_image(self, 
                          mz_value: float, 
                          tol: float = 0.1, 
                          reduce_func: Callable[[numpy.ndarray], float] | None = sum) -> Image:
        """Returns an ion image for the given mz_value. Reimplentation of `getionimage` from pyimzml.

        Args:
            mz_value (float): m/z value to extract.
            tol (float, optional): tolerance to extract mz_value. Defaults to 0.1.
            reduce_func (Callable[[numpy.ndarray], float], optional): Function to accumulate intensities that fall into [mz_value - tol, mz_value + tol]. Defaults to sum.

        Returns:
            Image: Computed ion image.
        """
        spec_to_intensity = {}
        for idx, (_, _, _) in enumerate(self.msi.coordinates):
            mzs, intensities = self.msi.getspectrum(idx)
            ints_ = intensities[np.logical_and((mz_value - tol) < mzs, (mz_value + tol) > mzs)]
            val = reduce_func(ints_)
            spec_to_intensity[idx] = val
        # local_idx_measurement_dict = {x: table[x].to_numpy() for x in table}
        rev_spec_to_ref_map = self.get_spec_to_ref_map(reverse=True)
        composed_dict = compose_dicts(rev_spec_to_ref_map, spec_to_intensity)
        composed_dict[self.background] = 0
        indexer = np.array([composed_dict.get(i) for i in range(self.ref_mat.data.min(), self.ref_mat.data.max() + 1)])
        ion_image = indexer[(self.ref_mat.data - self.ref_mat.data.min())]
        return Image(data=ion_image)
    
    def extract_ion_image_by_idx(self,
                                 index: int) -> Image:   
        """Extracts ion image by indexing the intensity array. Useful when data has been preprocessed and only contains a few ions.

        Args:
            index (int):

        Returns:
            Image:
        """
        spec_to_intensity = {}
        for idx, (_, _, _) in enumerate(self.msi.coordinates):
            _, intensities = self.msi.getspectrum(idx)
            spec_to_intensity[idx] = intensities[index]
        rev_spec_to_ref_map = self.get_spec_to_ref_map(reverse=True)
        composed_dict = compose_dicts(rev_spec_to_ref_map, spec_to_intensity)
        composed_dict[self.background] = 0
        indexer = np.array([composed_dict.get(i) for i in range(self.ref_mat.data.min(), self.ref_mat.data.max() + 1)])
        ion_image = indexer[(self.ref_mat.data - self.ref_mat.data.min())]
        return Image(data=ion_image)

    def to_ion_images(self,
                      table: pandas.DataFrame, 
                      background_value: int = 0):
        """Produces ion images for the given table. Table columns should refer to the indices used to index msi spectra by pyimzml.

        Args:
            table (pandas.DataFrame): Table mapping each msi a pixel to a vector of analytes. 
                Should have the shape Analytes X Pixel. Table indices are used as labels.
            background_value (int, optional): Defaults to 0.

        Returns:
            Annotation: Ion images.
        """
        n_ints = table.shape[0]
        ref_mat = self.ref_mat.data
        ion_cube = np.zeros((ref_mat.shape[0], ref_mat.shape[1], n_ints))
        
        local_idx_measurement_dict = {x: table[x].to_numpy() for x in table}
        rev_spec_to_ref_map = self.get_spec_to_ref_map(reverse=True)
        composed_dict = compose_dicts(rev_spec_to_ref_map, local_idx_measurement_dict)
        composed_dict[background_value] = np.zeros(n_ints)
        indexer = np.array([composed_dict.get(i) for i in range(ref_mat.min(), ref_mat.max() + 1)])
        ion_cube = indexer[(ref_mat - ref_mat.min())]
        ion_cube_annotation = Annotation(data=ion_cube,
                                        labels=table.index.to_list())
        return ion_cube_annotation