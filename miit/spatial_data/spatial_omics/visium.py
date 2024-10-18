from dataclasses import dataclass, field
import json
import math
import os
from os.path import join, exists
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple
import uuid

import cv2
import numpy, numpy as np
import pandas, pandas as pd

from miit.spatial_data.base_types import Annotation, Image, Pointset
from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics
from miit.registerers.base_registerer import Registerer


def merge_dicts(dict1: dict, dict2: dict) -> dict:
    new_dict = {}
    for key in dict1:
        if dict1[key] in dict2:
            new_dict[key] = dict2[dict1[key]]
    return new_dict


def convert_table_to_mat(table: pandas.core.frame.DataFrame,
                         visium: 'Visium',
                         col: Any):
    """
    Utility function that uses a visium object to convert integrated data
    from a dataframe format to a matrix format by projecting onto the
    reference matrix.
    """
    return get_measurement_matrix_sep(table, 
                                      visium.ref_mat.data,
                                      visium.table.data,
                                      col)
    

def get_measurement_matrix_sep(measurement_df: pandas.core.frame.DataFrame, 
                               ref_mat: numpy.ndarray, 
                               st_table: pandas.core.frame.DataFrame, 
                               col: Any):
    local_idx_measurement_dict = get_measurement_dict(measurement_df, col)
    intern_idx_local_idx_dict = {}
    for idx, row in st_table.iterrows():
        intern_idx_local_idx_dict[int(row['int_idx'])] = idx
    merged_dict = merge_dicts(intern_idx_local_idx_dict, local_idx_measurement_dict)
    indexer = np.array([merged_dict.get(i, 0) for i in range(ref_mat.min(), ref_mat.max() + 1)])
    measurement_mat = indexer[(ref_mat - ref_mat.min())]
    return measurement_mat


def get_measurement_dict(df: pandas.core.frame.DataFrame, col1: Any):
    dct = {}
    for idx, row in df.iterrows():
        dct[idx] = row[col1]
    return dct


def get_scalefactor(scalefactors: dict, image_scale: str):
    # 1 corresponds to ogirinal image size.
    scalefactor = 1
    if image_scale == 'lowres':
        scalefactor = scalefactors['tissue_lowres_scalef']
    elif image_scale == 'hires':
        scalefactor = scalefactors['tissue_hires_scalef']
    return scalefactor
    

def scale_tissue_positions(tissue_positions: pandas.core.frame.DataFrame,
                           scalefactors: dict,
                           image_scale: str):
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
    config: Optional[dict] = None
    tissue_mask: Optional[numpy.ndarray] = None
    background: ClassVar[int] = 0
    
    def __post_init__(self):
        if not self.skip_ref_mat_creation:
            self.__init_ref_mat()
        self._id = uuid.uuid1()
    
    def __init_ref_mat(self):
        self.build_ref_mat()

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
    def load(cls, directory: str):
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
        

    def build_ref_mat(self):
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
        
    def pad(self, padding: Tuple[int, int, int, int]):
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
             **kwargs: Dict) -> 'Visium':
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

    @classmethod
    def from_spcrng(cls, 
                    directory: str,
                    image_scale: str = 'hires',
                    fullres_image_path: str = None,
                    config: Dict =None):
        """
        Initiates Visium10X from spaceranger output directory.
        
        directory: output directory of spaceranger, typically named out.
        """
        if not exists(directory):
            # Throw nice custom exception here.
            pass
        path_to_scalefactors = join(directory, 'spatial', 'scalefactors_json.json')
        path_to_tissue_positions = join(directory, 'spatial', 'tissue_positions_list.csv')
        if image_scale == 'lowres':
            path_to_image = join(directory, 'spatial', 'tissue_lowres_image.png')
        elif image_scale == 'hires':
            path_to_image = join(directory, 'spatial', 'tissue_hires_image.png')
        else:
            path_to_image = fullres_image_path
        return Visium.from_spcrng_files(path_to_scalefactors,
                                           path_to_tissue_positions,
                                           path_to_image,
                                           image_scale,
                                           config)
    
    @classmethod
    def from_spcrng_files(cls,
                          path_to_scalefactors: str,
                          path_to_tissue_positions: str,
                          path_to_image: str,
                          image_scale: str = 'hires',
                          config: Dict = None):
        """
        Loads Visium10X object from spaceranger output.
        
        path_to_scalefactors:
        path_to_tissue_positions:
        path_to_image:
        image_scale: One of 'lowres', 'highres', 'fullres'. Default is 'highres'. 
        """
        # Select right scaling from scalefactors based on the supplied image.
        if image_scale not in ['lowres', 'hires', 'fullres']:
            pass
            # Throw exception.
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
        

    def apply_tissue_mask(self, ref_mat: Annotation):
        if self.tissue_mask is not None:
            ref_mat.data *= self.tissue_mask