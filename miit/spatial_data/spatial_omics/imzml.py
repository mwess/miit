from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from os.path import join, exists
from typing import (
    Any, 
    ClassVar, 
    Dict, 
    Optional, 
    Tuple, 
    Set,
    List,
    Union
)
import uuid

from lxml import etree
import pandas, pandas as pd
import numpy, numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

import pyimzml
from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLWriter import ImzMLWriter

from miit.spatial_data.base_types import (
    Annotation,
    BaseImage,
    Image,
    read_image
)
from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics
from miit.registerers.base_registerer import Registerer
from miit.utils.utils import copy_if_not_none
from miit.utils.imzml_preprocessing import do_msi_registration


# TODO: Fix that function.
def to_ion_images(table: pandas.core.frame.DataFrame, 
                  imzml: 'Imzml', 
                  background_value: int = 0):
    """_summary_

    Args:
        table (pandas.core.frame.DataFrame): Table mapping each msi a pixel to a vector of analytes. Should have the shape Analytes X Pixel.
        imzml (Imzml): Imzml, defines the topology of the ion images.
        background_value (int, optional): Defaults to 0.

    Returns:
        Annotation: Ion images.
    """
    n_ints = table.shape[0]
    rev_ref_to_spec_map = imzml.get_spec_to_ref_map(reverse=True)
    ref_mat = imzml.ref_mat.data
    ion_cube = np.zeros((ref_mat.shape[0], ref_mat.shape[1], n_ints))
    for i in range(ref_mat.shape[0]):
        for j in range(ref_mat.shape[1]):
            val = ref_mat[i,j]
            if background_value == val:
                ion_cube[i,j,:] = 0
            else:
                imzml_idx = rev_ref_to_spec_map[val]
                ints = table.loc[:, imzml_idx].to_numpy()
                ion_cube[i,j] = ints
    ion_cube_annotation = Annotation(data=ion_cube, 
                                     labels=table.index.to_list())
    return ion_cube_annotation


def export_imzml(template_msi: pyimzml.ImzMLParser.ImzMLParser, 
                 output_path: str,
                 integrated_data: pandas.core.frame.DataFrame) -> None:
    """Exports integrated msi data into the imzML format. Most of the work is done by
    `pyimzml`'s `ImzMLWriter`. However some information such as pixel size is not provided, which
    we add by manually writing the information to the imzML file.

    
    Args:
        template_msi (pyimzml.ImzMLParser.ImzMLParser): Template msi. Defines target topology.
        output_path (str): 
        integrated_data (pandas.core.frame.DataFrame): DataFrame containing integrated msi data.
    """

    mzs = integrated_data.columns.to_numpy()
    if mzs.dtype != np.float64:
        mzs = mzs.astype(np.float64)
    with ImzMLWriter(output_path, mz_dtype=np.float64, intensity_dtype=np.float64) as writer:
        for i, coords in enumerate(template_msi.coordinates):
            if i not in integrated_data.index:
                continue
            intensities = integrated_data.loc[i].to_numpy()
            if intensities.dtype != np.float64:
                intensities = intensities.astype(np.float64)
            writer.addSpectrum(mzs, intensities, coords)
    # Now we add additional parameters that imzml skipped.
    scan_settings_params = [
        ("max dimension x", "IMS:1000044", template_msi.imzmldict['max dimension x']),
        ("max dimension y", "IMS:1000045", template_msi.imzmldict['max dimension y']),
        ("pixel size x", "IMS:1000046", template_msi.imzmldict['pixel size x']),
        ("pixel size y", "IMS:1000047", template_msi.imzmldict['pixel size y']),
    ]
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
    return intensities - np.median(intensities[:100])


def find_nearest(array: numpy.ndarray, value: float) -> Tuple[float, int]:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def tic_trapz(intensity: float, 
              intensities: numpy.ndarray, 
              mz: Optional[float] = None) -> numpy.ndarray:
    return np.array(intensity) / trapezoid(y=intensities, x=mz)


def get_metabolite_intensities(
        msi: pyimzml.ImzMLParser.ImzMLParser, 
        mz_dict: Dict, 
        spectra_idxs: Set[int]) -> Dict[Union[int, str], List[float]]:
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
        for key in mz_dict:
            mz_elem = mz_dict[key]
            mz, radius = mz_elem['interval']
            lower_bound = find_nearest(mzs, mz * (1 - radius))
            upper_bound = find_nearest(mzs, mz * (1 + radius))                    
            intensity = norm_f(intensity_f(intensities[lower_bound[1]:upper_bound[1]]), intensities)        
            intensities_per_spot[spectrum_idx].append(intensity)
    return intensities_per_spot


def get_metabolite_intensities_from_full_spectrum(msi: pyimzml.ImzMLParser.ImzMLParser,
                                                  spectra_idxs: List[int], 
                                                  mz_intervals: Tuple[float, float, float],
                                                  norm_f: Optional[callable] = None,
                                                  baseline_f: Optional[callable] = None) -> Dict:
    """
    Identifies intensity peaks based on the list of provided `mz_intervals` in `msi`. `baseline_f` can be
    used to preprocess intensities, `norm_f` is used to determine the intensity value within the given mz_interval. 
    Only spectra within `spectra_idxs` will be processed.
    
    Returns:
        Dictionary of spectra_idxs to intensity peaks.
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


def get_metabolite_intensities_preprocessed(msi: pyimzml.ImzMLParser.ImzMLParser,
                                            spectra_idxs: Set[int],
                                            mz_intervals: Optional[List[Dict]] = None) -> Dict[Union[int, str], List[float]]:
    """Extracts metabolites from imzml file. Assumes that targets have been preprocessed and selected in SCiLS prior to exporting."""
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
                                        spectra_idxs: Set[int],
                                        mz_labels=None) -> Tuple[Dict[Union[int, str], List[float]], List[str]]:
    collected_intensities = {}
    for spectrum_idx in spectra_idxs:
        mzs, intensities = msi.getspectrum(spectrum_idx)
        collected_intensities[spectrum_idx] = intensities.copy()
    if mz_labels is None:
        mz_labels = ["{:10.3f}".format(x).strip() for x in mzs]
    metabolite_df = pd.DataFrame(collected_intensities, index=mz_labels)
    return metabolite_df


def convert_to_matrix(msi: pyimzml.ImzMLParser.ImzMLParser, 
                      srd: dict = None, 
                      target_resolution: int = 1) -> Union[numpy.array, dict, Optional[numpy.array]]:
    """Computes reference matrix from msi scaled to target_resolution. Will also convert a srd annotation to
    a binary annotation matrix, if provided. 

    Args:
        msi (pyimzml.ImzMLParser.ImzMLParser): Source imzml.
        srd (dict, optional): Additional srd annotation (from SCiLS). Defaults to None.
        target_resolution (int, optional): Target resolution to which. Defaults to 1.


    Returns:
        Union[numpy.array, dict, Optional[numpy.array]]: Reference matrix, mapping of msi pixels to reference matrix, If supplied, srd in matrix form.
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


def compute_mean_spectrum(msi: pyimzml.ImzMLParser.ImzMLParser):
    total_intensities = None
    for i in range(len(msi.coordinates)):
        mzs, intensities = msi.getspectrum(i)
        if total_intensities is None:
            total_intensities = intensities.copy()
        else:
            total_intensities += intensities
    avg_spec = total_intensities/len(msi.coordinates)
    return avg_spec


def load_metabolites(table_path: str, 
                     imzml_path: str) -> Tuple[dict, pandas.core.frame.DataFrame]:
    # NEDC_peak_table = pd.read_csv('Peaklist_136_NEDC_figshare.txt', sep='\t')
    NEDC_peak_table = pd.read_csv(table_path, sep='\t')
    NEDC_peak_table_IDed = NEDC_peak_table[NEDC_peak_table['ID'].notna()][['m/z', 'ID', 'ID in OPLSDA']].reset_index(drop=True)
    msi = ImzMLParser(imzml_path)
    peaks = get_peaks(msi, rel_percentage=0.00005)
    peak_dict, peak_intervals = get_one_peak_dict_and_interval_list(find_ided_peaks(peaks, NEDC_peak_table_IDed))
    return peak_dict, NEDC_peak_table_IDed


def get_peaks(msi: pyimzml.ImzMLParser.ImzMLParser, rel_percentage=0.00025):
    mzs = msi.getspectrum(0)[0]
    mean_intensities = compute_mean_spectrum(msi)
    norm_intensities = mean_intensities / trapezoid(y=mean_intensities, x=None)
    norm_intensities = 100 * norm_intensities /  norm_intensities.max()
    peaks, _ = find_peaks(norm_intensities, height=rel_percentage)
    p_m = [(mean_intensities, mzs)]
    return p_m


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


def get_one_peak_dict_and_interval_list(peaks_id: Tuple[int, float, float], 
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


def compute_weighted_average(measurements: Union[pandas.core.frame.DataFrame, numpy.ndarray],
                             weights: numpy.ndarray, 
                             background_weight: float) -> pandas.core.frame.DataFrame:
    return (measurements*weights).sum()/(sum(weights) + background_weight)


def msi_default_accumulate_spot_weighted_mean(source_keys: numpy.ndarray,
                                              source_counts: numpy.ndarray,
                                              measurement_df: pandas.core.frame.DataFrame,
                                              bck_weight: float) -> pandas.core.frame.DataFrame:
    selected_datas = measurement_df[source_keys].transpose()
    return pd.DataFrame(selected_datas.apply(lambda x: compute_weighted_average(x, source_counts, bck_weight), axis=0)).transpose()
  

def msi_default_spot_accumulation_fun(source_keys: numpy.ndarray, 
                                      source_counts: numpy.ndarray, 
                                      measurement_df: pandas.core.frame.DataFrame, 
                                      bck_weight: float,
                                      accumulator_function: Optional[callable] = None) -> pandas.core.frame.DataFrame:
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


def flatten_to_row(df: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
    v = df.unstack().to_frame().sort_index(level=1).T
    v.columns = v.columns.map('_'.join)    
    return v


@dataclass
class Imzml(BaseSpatialOmics):

    image: BaseImage 
    __ref_mat: Annotation = field(init=False, default=None)
    spec_to_ref_map: dict
    ann_mat: Optional[Annotation] = None
    background: ClassVar[int] = 0
    config: Optional[dict] = None
    name: str = ''

    def __post_init__(self):
        self._id = uuid.uuid1()

    @property
    def ref_mat(self):
        return self.__ref_mat
    
    @ref_mat.setter
    def ref_mat(self, ref_mat: Annotation):
        self.__ref_mat = ref_mat

    @staticmethod
    def get_type() -> str:
        return 'Imzml'

    def pad(self, padding: Tuple[int, int, int, int]):
        self.image.pad(padding)
        self.__ref_mat.pad(padding, constant_values=self.background)
        if self.ann_mat is not None:
            self.ann_mat.pad(padding, constant_values=0)

    def resize(self, height: int, width: int):
        self.image.resize(height, width)
        self.__ref_mat.resize(height, width)
        if self.ann_mat is not None:
            self.ann_mat.resize(height, width)

    def rescale(self, scaling_factor: float):
        self.image.rescale(scaling_factor)
        self.__ref_mat.rescale(scaling_factor)
        if self.ann_mat is not None:
            self.ann_mat.rescale(scaling_factor)

    def crop(self, x1: int, x2: int, y1: int, y2: int):
        self.image.crop(x1, x2, y1, y2)
        self.__ref_mat.crop(x1, x2, y1, y2)
        if not self.ann_mat is None:
            self.ann_mat.crop(x1, x2, y1, y2)

    def get_spec_to_ref_map(self, reverse: bool = False):
        map_ = None
        if reverse:
            map_ = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        else:
            map_ = self.spec_to_ref_map.copy()
        return map_

    def copy(self):
        ref_mat = self.__ref_mat.copy()
        ann_mat = copy_if_not_none(self.__ref_mat)
        spec_to_ref_map = self.spec_to_ref_map.copy()
        obj = Imzml(
            config=self.config.copy(),
            image=self.image.copy(),
            spec_to_ref_map=spec_to_ref_map,
            ann_mat=ann_mat,
        )
        obj.ref_mat = ref_mat
        return obj

    def apply_transform(self, 
             registerer: Registerer, 
             transformation: Any, 
             **kwargs: Dict) -> 'Imzml':
        image_transformed = self.image.apply_transform(registerer, transformation, **kwargs)
        ref_mat_transformed = self.__ref_mat.apply_transform(registerer, transformation, **kwargs)
        if self.ann_mat is not None:
            ann_mat_transformed = self.ann_mat.apply_transform(registerer, transformation, **kwargs)
        else:
            ann_mat_transformed = None
        config = self.config.copy() if self.config is not None else None
        scils_export_imzml_transformed = Imzml(
            config=config,
            image=image_transformed, 
            spec_to_ref_map=self.spec_to_ref_map,
            ann_mat=ann_mat_transformed,
            name=self.name)
        scils_export_imzml_transformed.ref_mat = ref_mat_transformed
        return scils_export_imzml_transformed

    def flip(self, axis: int = 0):
        self.image.flip(axis=axis)
        self.__ref_mat.flip(axis=axis)
        if self.ann_mat is not None:
            self.ann_mat.flip(axis=axis)

    def store(self, directory: str):
        if not exists(directory):
            os.mkdir(directory)
        directory = join(directory, str(self._id))
        if not exists(directory):
            os.mkdir(directory)
        f_dict = {}
        config_path = join(directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
        f_dict['config_path'] = config_path
        self.image.store(join(directory, 'image'))
        f_dict['image'] = join(directory, 'image')
        self.__ref_mat.store(join(directory, 'ref_mat'))
        f_dict['__ref_mat'] = join(directory, 'ref_mat')
        spec_to_ref_map_path = join(directory, 'spec_to_ref_map.json')
        with open(spec_to_ref_map_path, 'w') as f:
            json.dump(self.spec_to_ref_map, f)
        f_dict['spec_to_ref_map_path'] = spec_to_ref_map_path
        if self.ann_mat is not None:
            self.ann_mat.store(join(directory, str(self.ann_mat._id)))
            f_dict['ann_mat'] = join(directory, str(self.ann_mat._id))
        f_dict['name'] = self.name
        with open(join(directory, 'attributes.json'), 'w') as f:
            json.dump(f_dict, f)

    @classmethod
    def load(cls, directory: str) -> 'Imzml':
        with open(join(directory, 'attributes.json')) as f:
            attributes = json.load(f)
        with open(attributes['config_path']) as f:
            config = json.load(f)
        image = Image.load(attributes['image'])
        __ref_mat = Annotation.load(attributes['__ref_mat'])
        with open(attributes['spec_to_ref_map_path']) as f:
            spec_to_ref_map = json.load(f)
        ann_mat_path = attributes.get('ann_mat', None)
        if ann_mat_path is not None:
            ann_mat = Annotation.load(ann_mat_path)
        else:
            ann_mat = None
        name = attributes.get('name', '')
        obj = cls(
            config=config,
            image=image,
            spec_to_ref_map=spec_to_ref_map,
            ann_mat=ann_mat,
            name=name
        ) 
        obj.ref_mat = __ref_mat
        id_ = uuid.UUID(os.path.basename(directory.strip('/')))
        obj._id = id_
        return obj
        
    @classmethod
    def load_msi_data(cls, 
                      image: Union[BaseImage, numpy.array], 
                      imzml_path: str, 
                      name: str = '',
                      config: Optional[Dict] = None,
                      srd_path: Optional[str] = None,
                      use_srd: bool = False,
                      enable_msi_registration: bool = True,
                      registerer: Optional[Registerer] = None,
                      ref_image_resolution: Optional[int] = 1):
        msi = ImzMLParser(imzml_path)
        if config is None:
            config = {}
        if 'imzml' not in config:
            config['imzml'] = imzml_path
        if srd_path is not None:
            if 'srd' not in config:
                config['srd'] = srd_path
            with open(srd_path, 'rb') as f:
                srd = json.load(f)
        else:
            srd = None
        ref_mat, spec_to_ref_map, ann_mat = convert_to_matrix(msi, srd, target_resolution=ref_image_resolution)
        if enable_msi_registration:
            if ann_mat is not None:
                additional_images = [ann_mat]
            else:
                additional_images = []
            if use_srd:
                reg_img = ann_mat
            else:
                reg_img = None
            if isinstance(image, numpy.ndarray):
                image = Image(data=image)
            _, ref_mat, add_imgs = do_msi_registration(image.data, 
                                                       ref_mat, 
                                                       spec_to_ref_map, 
                                                       msi, 
                                                       reg_img=reg_img,
                                                       additional_images=additional_images,
                                                       registerer=registerer)
            if ann_mat is not None:
                ann_mat = add_imgs[0]
        ref_mat = ref_mat.astype(int)
        ref_mat = Annotation(data=ref_mat)
        if ann_mat is not None:
            ann_mat = Annotation(data=ann_mat)
        else:
            ann_mat = None
        obj = cls(
            config=config,
            image=image,
            spec_to_ref_map=spec_to_ref_map,
            ann_mat=ann_mat,
            name=name
        )
        obj.ref_mat = ref_mat
        return obj
            
    
    @classmethod
    def from_config(cls, config):
        # TODO: Maybe add resolution to config
        imzml_path = config['imzml']
        image_path = config['image']
        image = read_image(image_path)
        resolution = config.get('resolution', 1)
        msi = ImzMLParser(imzml_path)
        srd_path = config.get('srd', None)
        if srd_path is not None:
            with open(srd_path, 'rb') as f:
                srd = json.load(f)
        else:
            srd = None
        ref_mat, spec_to_ref_map, ann_mat = convert_to_matrix(msi, srd, target_resolution=resolution)
        do_pre_registration = config.get('preregistration', True)
        if do_pre_registration:
            if ann_mat is not None:
                additional_images = [ann_mat]
            else:
                additional_images = []
            use_srd_ann = config.get('use_srd_ann', False)
            if use_srd_ann:
                reg_img = ann_mat
            else:
                reg_img = None
            _, ref_mat, add_imgs = do_msi_registration(image.data, 
                                                       ref_mat, 
                                                       spec_to_ref_map, 
                                                       msi, 
                                                       reg_img=reg_img,
                                                       additional_images=additional_images)
            if ann_mat is not None:
                ann_mat = add_imgs[0]
        ref_mat = ref_mat.astype(int)
        ref_mat = Annotation(data=ref_mat)
        if ann_mat is not None:
            ann_mat = Annotation(data=ann_mat)
        else:
            ann_mat = None
        obj = cls(
            config=config,
            image=image,
            spec_to_ref_map=spec_to_ref_map,
            ann_mat=ann_mat
        )
        obj.ref_mat = ref_mat
        return obj

    def convert_mappings_and_unique_ids_back(self, 
                                             mappings: dict, 
                                             unique_ids: set) -> Tuple[dict, set]:
        for key in mappings:
            mappings[key] = mappings[key] - 1
        unique_ids = {x - 1 for x in unique_ids}
        return mappings, unique_ids

    def set_map_to_msi_pixel_idxs(self, ref_mat_values: Optional[Set] = None) -> Set:
        # Invert map
        spec_to_ref_map_rev = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        if ref_mat_values is None:
            ref_mat_values = spec_to_ref_map_rev.keys()
        return {int(spec_to_ref_map_rev[x]) for x in ref_mat_values}

    def mappings_map_to_msi_pixel_idxs(self, mappings: dict) -> dict:
        spec_to_ref_map_rev = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        mapped_mappings = {}
        for key in mappings:
            idx_arr = mappings[key][0]
            idx_arr_mapped = np.array([int(spec_to_ref_map_rev[x]) for x in idx_arr])
            mapped_mappings[key] = (idx_arr_mapped, mappings[key][1].copy())
        return mapped_mappings

    def spots_background_map_keys_to_msi_pixel_idxs(self, spots_background: dict) -> dict:
        # TODO: Remove that function again.
        spec_to_ref_map_rev = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        return {spec_to_ref_map_rev[x]: spots_background[x] for x in spots_background}