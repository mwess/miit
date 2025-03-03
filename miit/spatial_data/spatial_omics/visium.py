"""
Module that handles Visium data.
"""
from dataclasses import dataclass, field
import json
import math
import os
from os.path import join, exists
from pathlib import Path
from typing import Any, ClassVar
import uuid

import cv2
import h5py
import numpy, numpy as np
import pandas, pandas as pd

from miit.spatial_data.base_types import Annotation, Image, Pointset
from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics
from miit.registerers.base_registerer import Registerer


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


# TODO: Add example for row_features.
def load_visium_data_matrix(path: str,
                            row_feature: list[str] | str | None = None) -> pandas.DataFrame:
    """ Reads Visium data from SpaceRanger output h5 file and convert to a pandas DataFrane. The dataframe
    has the shape (feature, barcode).

    Args:
        path (str): Path to h5 file.
        row_feature: Feature name to name rows. If None, features are enumerate. If row_feature
                     is a list, features are joined by ' - '.
    
    Returns:
        pandas.DataFrame: Visium data. 
    """
    f = h5py.File(path, 'r')
    indptr = f['matrix']['indptr'][:]
    indices = f['matrix']['indices'][:]

    ys = np.array(range(indptr.shape[0]-1))
    indptr_diffs = indptr[1:] - indptr[:-1]
    y_inds = np.repeat(ys, indptr_diffs)

    mat = np.zeros(f['matrix']['shape'][:2])
    data = f['matrix']['data'][:]
    mat[indices, y_inds] = data

    # Get barcodes
    barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes'][:]]

    # Get row names
    if row_feature is None:
        row_names = list(range(mat.shape[0]))
    elif isinstance(row_feature, str):
        row_names = [x.decode('utf-8') for x in f['matrix']['features'][row_feature][:]]
    elif isinstance(row_feature, list):
        r_feat_list = []
        for row_feature_ in row_feature:
            row_names_ = [x.decode('utf-8') for x in f['matrix']['features'][row_feature_][:]]
            r_feat_list.append(row_names_)
        row_names = [' - '.join(names) for names in zip(*r_feat_list)]

    df = pd.DataFrame(mat, columns=barcodes, index=row_names)
    return df


def convert_table_to_mat(measurement_df: pandas.DataFrame,
                         visium: 'Visium',
                         col: Any) -> numpy.ndarray:
    """Utility function that uses a visium object to convert integrated data
    from a dataframe format to a matrix format by projecting onto the
    reference matrix.

    Args:
        measurement_df (pandas.DataFrame): Table containing measurements.
        visium (Visium): 
        col (Any): Column in measurement_df.

    Returns:
        numpy.ndarray: 
    """
    return fill_measurement_matrix(measurement_df, 
                                      visium.ref_mat.data,
                                      visium.table.data,
                                      col)
    

def fill_measurement_matrix(measurement_df: pandas.DataFrame, 
                            ref_mat: numpy.ndarray, 
                            st_table: pandas.DataFrame, 
                            col: Any) -> numpy.ndarray:
    """Fill the layout defined by ref_mat with values from ST data. 

    Args:
        measurement_df (pandas.DataFrame): 
        ref_mat (numpy.ndarray): 
        st_table (pandas.DataFrame): 
        col (Any): 

    Returns:
        numpy.ndarray: 
    """
    local_idx_measurement_dict = get_measurement_dict(measurement_df, col)
    intern_idx_local_idx_dict = {int(row['int_idx']): idx for idx, row in st_table.iterrows()}
    comp_dict = compose_dicts(intern_idx_local_idx_dict, local_idx_measurement_dict)
    indexer = np.array([comp_dict.get(i, 0) for i in range(ref_mat.min(), ref_mat.max() + 1)])
    measurement_mat = indexer[(ref_mat - ref_mat.min())]
    return measurement_mat


def get_measurement_dict(df: pandas.DataFrame, key_col: Any) -> dict:
    """Converts a dataframe to a dicionary.
    
    Args:
        df (pandas.DataFrame): Source dataframe.
        key_col (Any): Column to use as an index for the dataframe.
    """
    return {idx: row[key_col] for idx, row in df.iterrows()}


def get_scalefactor(scalefactors: dict, image_scale: str) -> float:
    """Get scalefactors from ST data.

    Args:
        scalefactors (dict): 
        image_scale (str): 

    Returns:
        float: 
    """
    # 1 corresponds to ogirinal image size.
    scalefactor = 1
    if image_scale == 'lowres':
        scalefactor = scalefactors['tissue_lowres_scalef']
    elif image_scale == 'hires':
        scalefactor = scalefactors['tissue_hires_scalef']
    return scalefactor
    

def scale_tissue_positions(tissue_positions: pandas.DataFrame,
                           scalefactors: dict,
                           image_scale: str) -> pandas.DataFrame:
    """
    Scales tissue position according to new scale.
    
    Args:
        tissue_positions (pandas.DataFrame):
        scale_factors (dict):
        image_scale (str): Key for scale_factors.
        
    Returns:
        pandas.DataFrame: 
    """
    scale = 1
    if image_scale == 'lowres':
        scale = scalefactors['tissue_lowres_scalef']
    elif image_scale == 'hires':
        scale = scalefactors['tissue_hires_scalef']
    tissue_positions['x'] = tissue_positions['x'] * scale
    tissue_positions['y'] = tissue_positions['y'] * scale
    return tissue_positions


@dataclass
class Visium(BaseSpatialOmics):
    
    image: Image
    table: Pointset
    scale_factors: dict
    __ref_mat: Annotation = field(init=False)
    spec_to_ref_map: dict = field(init=False)
    name: str = ''
    skip_ref_mat_creation: bool = False
    config: dict | None = None
    tissue_mask: numpy.ndarray | None = None
    background: ClassVar[int] = 0
    
    def __post_init__(self):
        if not self.skip_ref_mat_creation:
            self.__init_ref_mat()
        self._id = uuid.uuid1()
    
    def __init_ref_mat(self):
        self._build_ref_mat()

    @property
    def ref_mat(self):
        return self.__ref_mat
    
    @ref_mat.setter
    def ref_mat(self, ref_mat: Annotation):
        self.__ref_mat = ref_mat

    def store(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        f_dict = {}
        image_path = join(directory, 'image')
        if not exists(image_path):
            os.mkdir(image_path)
        self.image.store(image_path)
        # f_dict['image'] = join(image_path, str(self.image._id))
        f_dict['image'] = image_path
        table_path = join(directory, 'table')
        if not exists(table_path):
            os.mkdir(table_path)
        self.table.store(table_path)
        # f_dict['table'] = join(table_path, str(self.table._id))
        f_dict['table'] = table_path 
        scale_factors_path = join(directory, 'scale_factors.json')
        with open(scale_factors_path, 'w') as f:
            json.dump(self.scale_factors, f)
        f_dict['scale_factors_path'] = scale_factors_path
        if self.__ref_mat is not None:
            ref_mat_path = join(directory, 'ref_mat')
            if not exists(ref_mat_path):
                os.mkdir(ref_mat_path)
            self.__ref_mat.store(ref_mat_path)
            f_dict['__ref_mat'] = ref_mat_path 
        spec_to_ref_map_path = join(directory, 'spec_to_ref_mat.json')
        with open(spec_to_ref_map_path, 'w') as f:
            json.dump(self.spec_to_ref_map, f)
        f_dict['spec_to_ref_map_path'] = spec_to_ref_map_path
        if self.config is not None:
            config_path = join(directory, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f)
            f_dict['config_path'] = config_path
        f_dict['name'] = self.name
        f_dict['id'] = str(self._id)
        with open(join(directory, 'attributes.json'), 'w') as f:
            json.dump(f_dict, f)

    @classmethod
    def load(cls, directory: str) -> 'Visium':
        attributes_path = join(directory, 'attributes.json')
        with open(attributes_path) as f:
            attributes = json.load(f)
        image = Image.load(attributes['image'])
        table = Pointset.load(attributes['table'])
        sf_path = attributes['scale_factors_path']
        with open(sf_path) as f:
            scale_factors = json.load(f)
        if '__ref_mat' in attributes:
            __ref_mat = Annotation.load(attributes['__ref_mat'])
        else:
            __ref_mat = None
        spec_to_ref_map_path = attributes.get('spec_to_ref_map_path', None)
        if spec_to_ref_map_path is not None:
            with open(spec_to_ref_map_path) as f:
                spec_to_ref_map = json.load(f)
        else:
            spec_to_ref_map = None
        config_path = attributes.get('config_path', None)
        if config_path is not None:
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = None
        id_ = uuid.UUID(attributes['id'])
        name = attributes['name']
        obj = cls(
            image=image,
            table=table,
            scale_factors=scale_factors,
            skip_ref_mat_creation=True,
            config=config,
            name=name
        )
        obj.ref_mat = __ref_mat
        obj.spec_to_ref_map = spec_to_ref_map
        obj._id = id_
        return obj
        

    def _build_ref_mat(self):
        scalefactor = self.config['scalefactor']
        spot_diameter = int(round(self.scale_factors['spot_diameter_fullres'] * scalefactor))
        spot_radius = spot_diameter//2
        
        # Add numeric indices for the ref_mat
        self.spec_to_ref_map = {}
        ref_mat_idx_counter = 1
        for tbl_idx, _ in self.table.data.iterrows():
            self.spec_to_ref_map[tbl_idx] = ref_mat_idx_counter
            ref_mat_idx_counter += 1
        self.table.data['int_idx'] = range(1, self.table.data.shape[0] + 1)
        ref_mat = np.zeros((self.image.data.shape[0], self.image.data.shape[1]), dtype=np.int64)
        for tbl_idx, row in self.table.data.iterrows():
            x, y = row['x'], row['y']
            int_idx = self.spec_to_ref_map[tbl_idx]
            xl = max(math.floor(x - spot_radius), 0)
            xh = min(math.ceil(x + spot_radius), self.image.data.shape[0])
            yl = max(math.floor(y - spot_radius), 0)
            yh = min(math.ceil(y + spot_radius), self.image.data.shape[1])

            for i in range(xl, xh):
                for j in range(yl, yh):
                    if np.sqrt((i - x)**2 + (j - y)**2) <= spot_radius:
                        ref_mat[i,j] = int_idx
        self.__ref_mat = Annotation(data=ref_mat)

    @staticmethod
    def get_type() -> str:
        return 'visium'
        
    def pad(self, padding: tuple[int, int, int, int]):
        self.image.pad(padding)
        self.table.pad(padding)
        self.__ref_mat.pad(padding)

    def resize(self, height: int, width: int):
        h_old, w_old = self.image.data.shape[0], self.image.data.shape[1]
        self.image.resize(height, width)
        width_scale = width / w_old
        height_scale = height / h_old
        self.table.resize(height_scale, width_scale)
        self.__ref_mat.resize(height, width) 

    def rescale(self, scaling_factor: float):
        self.image.rescale(scaling_factor)
        self.table.rescale(scaling_factor)
        self.__ref_mat.rescale(scaling_factor)

    def crop(self, x1: int, x2: int, y1: int, y2: int):
        self.image.crop(x1, x2, y1, y2)
        self.table.crop(x1, x2, y1, y2)
        # TODO: Should points that are out-of-bounds of the image be filtered as well?
        if self.__ref_mat is not None:
            self.__ref_mat.crop(x1, x2, y1, y2)

    def flip(self, axis: int = 0):
        self.image.flip(axis=axis)
        self.table.flip(self.image.data.shape, axis=axis)
        if self.__ref_mat is not None:
            self.__ref_mat.flip(axis=axis)
        if self.tissue_mask is not None:
            self.tissue_mask.flip(axis=axis)

    def copy(self):
        if self.__ref_mat is not None:
            ref_mat = self.__ref_mat.copy()
        else:
            ref_mat = None
        obj = Visium(
            image=self.image.copy(),
            table=self.table.copy(),
            scale_factors=self.scale_factors.copy(),
            skip_ref_mat_creation=True,
            config=self.config.copy(),
        )
        obj.spec_to_ref_map = self.spec_to_ref_map
        obj.ref_mat = ref_mat
        return obj

    def apply_transform(self, 
             registerer: Registerer, 
             transformation: Any, 
             **kwargs: dict) -> 'Visium':
        image_transformed = self.image.apply_transform(registerer, transformation, **kwargs)
        ref_mat_warped = self.__ref_mat.apply_transform(registerer, transformation, **kwargs)
        ref_mat_warped = Annotation(data=ref_mat_warped.data)
        table = self.table.apply_transform(registerer, transformation, **kwargs)
        config = self.config.copy() if self.config is not None else None
        transformed_st_data = Visium(
            image=image_transformed, 
            table=table, 
            scale_factors=self.scale_factors, 
            skip_ref_mat_creation=True,
            config=config,
        )
        transformed_st_data.ref_mat = ref_mat_warped
        transformed_st_data.spec_to_ref_map = self.spec_to_ref_map.copy()
        return transformed_st_data

    def get_spec_to_ref_map(self, reverse: bool = False):
        map_ = None
        if reverse:
            map_ = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        else:
            map_ = self.spec_to_ref_map.copy()
        return map_
    
    def get_visium_data(self, 
                        name: str = 'filtered_feature_bc_matrix',
                        directory: str | None = None,
                        row_feature: str | list[str] | None = None):
        if not name.endswith('.h5'):
            name = f'{name}.h5'
        if name not in ['filtered_feature_bc_matrix.h5', 'raw_feature_bc_matrix.h5']:
            raise Exception(f'Invalid name supplied: {name}.')
        if directory is None and self.config is not None and 'directory' in self.config:
            directory = self.config('directory')
        path = join(directory, name)
        data = load_visium_data_matrix(path=path, row_feature=row_feature)
        return data

    @classmethod
    def from_spcrng(cls, 
                    directory: str,
                    image_scale: str = 'hires',
                    fullres_image_path: str = None,
                    config: dict =None) -> 'Visium':
        """Initiates Visium10X from spaceranger output directory.

        Args:
            directory (str): output directory of spaceranger, typically named out.
            image_scale (str, optional): Image scale used to st spots. Defaults to 'hires'.
            fullres_image_path (str, optional): Path to image if 'fullres' image scale is used.. Defaults to None.
            config (dict, optional): _description_. Defaults to None.

        Returns:
            Visium: _description_
        """
        if not exists(directory):
            # TODO: Throw nice custom exception here.
            pass
        path_to_scalefactors = join(directory, 'spatial', 'scalefactors_json.json')
        path_to_tissue_positions = join(directory, 'spatial', 'tissue_positions_list.csv')
        if image_scale == 'lowres':
            path_to_image = join(directory, 'spatial', 'tissue_lowres_image.png')
        elif image_scale == 'hires':
            path_to_image = join(directory, 'spatial', 'tissue_hires_image.png')
        else:
            path_to_image = fullres_image_path
        if config is None:
            config = {}
        config['directory'] = directory
        config['image_scale'] = image_scale
        return Visium.from_spcrng_files(path_to_scalefactors,
                                           path_to_tissue_positions,
                                           path_to_image,
                                           image_scale,
                                           config)
    
    # TODO: Check that it works with other version of ST.
    @classmethod
    def from_spcrng_files(cls,
                          path_to_scalefactors: str,
                          path_to_tissue_positions: str,
                          path_to_image: str,
                          image_scale: str = 'hires',
                          config: dict = None) -> 'Visium':
        """
        Loads Visium10X object from spaceranger output.
        
        Args:
            path_to_scalefactors (str):
            path_to_tissue_positions (str):
            path_to_image (str):
            image_scale (str): One of 'lowres', 'highres', 'fullres'. Default is 'highres'. 
        """
        # Select right scaling from scalefactors based on the supplied image.
        if image_scale not in ['lowres', 'hires', 'fullres']:
            raise Exception(f'Image scale invalid: {image_scale}')
        image = Image(data=cv2.imread(path_to_image))
        with open(path_to_scalefactors) as f:
            scalefactors = json.load(f)
        tissue_positions_df = pd.read_csv(path_to_tissue_positions, header=None, index_col=0)
        tissue_positions_df = tissue_positions_df.rename(columns={
            4: 'x',
            5: 'y'
        }, inplace=False)[['x', 'y']]
        tissue_positions_df = scale_tissue_positions(tissue_positions_df, scalefactors, image_scale)
        tissue_positions = Pointset(data=tissue_positions_df, index_col=0)
        if config is None:
            config = {}
        config['scalefactors'] = path_to_scalefactors
        config['tissue_positions_list'] = path_to_tissue_positions
        config['path_to_image'] = path_to_image
        config['image_scale'] = image_scale
        config['scalefactor'] = get_scalefactor(scalefactors, image_scale)
        return cls(image, tissue_positions, scalefactors, config=config)