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
from miit.spatial_data.base_classes import BaseImage, BaseSpatialOmics, MIITobject
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
    indptr = f['matrix']['indptr'][:] # type: ignore
    indices = f['matrix']['indices'][:] # type: ignore

    ys = np.array(range(indptr.shape[0]-1)) # type: ignore
    indptr_diffs = indptr[1:] - indptr[:-1] # type: ignore
    y_inds = np.repeat(ys, indptr_diffs)

    mat = np.zeros(f['matrix']['shape'][:2]) # type:ignore
    data = f['matrix']['data'][:] # type: ignore
    mat[indices, y_inds] = data

    # Get barcodes
    barcodes = [x.decode('utf-8') for x in f['matrix']['barcodes'][:]] # type: ignore

    # Get row names
    if row_feature is None:
        row_names = list(range(mat.shape[0]))
    elif isinstance(row_feature, str):
        row_names = [x.decode('utf-8') for x in f['matrix']['features'][row_feature][:]] # type: ignore
    elif isinstance(row_feature, list):
        r_feat_list = []
        for row_feature_ in row_feature:
            row_names_ = [x.decode('utf-8') for x in f['matrix']['features'][row_feature_][:]] # type: ignore
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


@MIITobject
@dataclass(kw_only=True)
class Visium(BaseSpatialOmics):
    
    image: BaseImage | None = None
    table: Pointset
    scale_factors: dict
    ref_mat: Annotation
    spec_to_ref_map: dict
    config: dict
    name: str = ''
    tissue_mask: Annotation | None = None
    background: ClassVar[int] = 0
    
    def __post_init__(self):
        self._id = uuid.uuid1()
    
    def store(self, directory: str):
        Path(directory).mkdir(parents=True, exist_ok=True)
        f_dict = {}
        if self.image is not None:
            image_path = join(directory, 'image')
            if not exists(image_path):
                os.mkdir(image_path)
            self.image.store(image_path)
            f_dict['image'] = image_path
        table_path = join(directory, 'table')
        if not exists(table_path):
            os.mkdir(table_path)
        self.table.store(table_path)
        f_dict['table'] = table_path 
        scale_factors_path = join(directory, 'scale_factors.json')
        with open(scale_factors_path, 'w') as f:
            json.dump(self.scale_factors, f)
        f_dict['scale_factors_path'] = scale_factors_path
        ref_mat_path = join(directory, 'ref_mat')
        if not exists(ref_mat_path):
            os.mkdir(ref_mat_path)
        self.ref_mat.store(ref_mat_path)
        f_dict['ref_mat'] = ref_mat_path 
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
        if 'image' in attributes:
            image = Image.load(attributes['image'])
        else:
            image = None
        table = Pointset.load(attributes['table'])
        sf_path = attributes['scale_factors_path']
        with open(sf_path) as f:
            scale_factors = json.load(f)
        ref_mat = Annotation.load(attributes['ref_mat'])
        spec_to_ref_map_path = attributes.get('spec_to_ref_map_path', None)
        with open(spec_to_ref_map_path) as f:
            spec_to_ref_map: dict = json.load(f)
        config_path = attributes.get('config_path', None)
        if config_path is not None:
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = {}
        id_ = uuid.UUID(attributes['id'])
        name = attributes['name']
        obj = cls(
            image=image,
            table=table,
            scale_factors=scale_factors,
            config=config,
            name=name,
            ref_mat=ref_mat,
            spec_to_ref_map=spec_to_ref_map
        )
        obj._id = id_
        return obj
        
    @staticmethod
    def build_ref_mat(scalefactor: float,
                      scale_factors: dict,
                      spot_positions: Pointset | pd.DataFrame,
                      image_shape: tuple[int, int]
                      ) -> tuple[Annotation, dict[int, int]]:
        spot_diameter = int(round(scale_factors['spot_diameter_fullres'] * scalefactor))
        spot_radius = spot_diameter//2
        
        # Add numeric indices for the ref_mat
        if isinstance(spot_positions, Pointset):
            spot_positions = spot_positions.data
        spec_to_ref_map = {}
        ref_mat_idx_counter = 1
        for tbl_idx, _ in spot_positions.iterrows():
            spec_to_ref_map[tbl_idx] = ref_mat_idx_counter
            ref_mat_idx_counter += 1
        spot_positions['int_idx'] = range(1, spot_positions.shape[0] + 1)
        ref_mat = np.zeros((image_shape[0], image_shape[1]), dtype=np.int64)
        for tbl_idx, row in spot_positions.iterrows():
            x: float = row['x'] # type: ignore
            y: float = row['y'] # type: ignore
            int_idx = spec_to_ref_map[tbl_idx]
            xl = max(math.floor(x - spot_radius), 0)
            xh = min(math.ceil(x + spot_radius), image_shape[0])
            yl = max(math.floor(y - spot_radius), 0)
            yh = min(math.ceil(y + spot_radius), image_shape[1])

            for i in range(xl, xh):
                for j in range(yl, yh):
                    if np.sqrt((i - x)**2 + (j - y)**2) <= spot_radius:
                        ref_mat[i,j] = int_idx
        ref_mat = Annotation(data=ref_mat)
        return ref_mat, spec_to_ref_map

    @staticmethod
    def get_type() -> str:
        return 'Visium'
        
    def pad(self, padding: tuple[int, int, int, int]):
        if self.image is not None:
            self.image.pad(padding)
        self.table.pad(padding)
        self.ref_mat.pad(padding)

    def resize(self, height: int, width: int):
        h_old, w_old = self.ref_mat.data.shape[0], self.ref_mat.data.shape[1]
        if self.image is not None:
            self.image.resize(height, width)
        width_scale = width / w_old
        height_scale = height / h_old
        self.table.resize(height_scale, width_scale)
        self.ref_mat.resize(height, width) 

    def rescale(self, scaling_factor: float):
        if self.image is not None:
            self.image.rescale(scaling_factor)
        self.table.rescale(scaling_factor)
        self.ref_mat.rescale(scaling_factor)

    # TODO: Add an option on whether to filter out points that are not in the image space after cropping.
    def crop(self, x1: int, x2: int, y1: int, y2: int):
        if self.image is not None:
            self.image.crop(x1, x2, y1, y2)
        self.table.crop(x1, x2, y1, y2)
        # TODO: Should points that are out-of-bounds of the image be filtered as well?
        self.ref_mat.crop(x1, x2, y1, y2)

    def flip(self, axis: int = 0):
        if self.image is not None:
            self.image.flip(axis=axis)
        self.table.flip(self.ref_mat.data.shape, axis=axis)
        self.ref_mat.flip(axis=axis)
        if self.tissue_mask is not None:
            self.tissue_mask.flip(axis=axis)

    def copy(self):
        config = self.config.copy() if self.config is not None else {}
        return Visium(
            image=self.image.copy() if self.image else None,
            table=self.table.copy(),
            scale_factors=self.scale_factors.copy(),
            config=config,
            ref_mat=self.ref_mat.copy(),
            spec_to_ref_map=self.spec_to_ref_map
        )

    def apply_transform(self, 
             registerer: Registerer, 
             transformation: Any, 
             **kwargs: dict) -> 'Visium':
        if self.image is not None:
            image_transformed = self.image.apply_transform(registerer, transformation, **kwargs)
        else:
            image_transformed = None
        ref_mat_warped = self.ref_mat.apply_transform(registerer, transformation, **kwargs)
        ref_mat_warped = Annotation(data=ref_mat_warped.data)
        table = self.table.apply_transform(registerer, transformation, **kwargs)
        config = self.config.copy() if self.config is not None else {}
        transformed_st_data = Visium(
            image=image_transformed, 
            table=table, 
            scale_factors=self.scale_factors, 
            config=config,
            ref_mat=ref_mat_warped,
            spec_to_ref_map=self.spec_to_ref_map.copy()
        )
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
                    fullres_image_path: str | None = None,
                    config: dict | None = None) -> 'Visium':
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
        if path_to_image is None:
            raise Exception('No path supplied.')
        return Visium.from_spcrng_files(path_to_scalefactors,
                                           path_to_tissue_positions,
                                           path_to_image,
                                           image_scale=image_scale,
                                           config=config)
    
    # TODO: Check that it works with other version of ST.
    @classmethod
    def from_spcrng_files(cls,
                          path_to_scalefactors: str,
                          path_to_tissue_positions: str,
                          path_to_image: str | None = None,
                          image_shape: tuple[int, int] | None = None,
                          image_scale: str = 'hires',
                          config: dict | None = None) -> 'Visium':
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
            # TODO: Throw exception
            pass

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
        config['path_to_image'] = path_to_image if path_to_image is not None else ''
        config['image_scale'] = image_scale
        config['scalefactor'] = get_scalefactor(scalefactors, image_scale)
        if path_to_image is not None:
            image = Image(data=cv2.imread(path_to_image)) # type:ignore
            image_shape = (image.data.shape[0], image.data.shape[1])
        else:
            image = None
            if image_shape is None:
                raise Exception('No image or image shape for reference matrix mapping was provided.')        
        ref_mat, spec_to_ref_map = Visium.build_ref_mat(
            scalefactor=config['scalefactor'],
            scale_factors=scalefactors,
            spot_positions=tissue_positions,
            image_shape=image_shape
        )
        return cls(image=image, 
                   table=tissue_positions, 
                   scale_factors=scalefactors, 
                   config=config,
                   ref_mat=ref_mat,
                   spec_to_ref_map=spec_to_ref_map)