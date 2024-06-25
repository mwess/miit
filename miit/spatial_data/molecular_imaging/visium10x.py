from dataclasses import dataclass, field
import json
import math
import os
from os.path import join, exists
from typing import Any, ClassVar, Dict, Optional, Tuple
import uuid

import cv2
import numpy
import numpy as np
import pandas
import pandas as pd

from miit.custom_types import PdDataframe
from miit.spatial_data.molecular_imaging.imaging_data import BaseMolecularImaging
from miit.registerers.base_registerer import Registerer
from miit.utils.utils import custom_max_voting_filter
from miit.spatial_data.image import Annotation, Image, Pointset


def get_scalefactor(scalefactors, image_scale):
    # 1 corresponds to ogirinal image size.
    scalefactor = 1
    if image_scale == 'lowres':
        scalefactor = scalefactors['tissue_lowres_scalef']
    elif image_scale == 'hires':
        scalefactor = scalefactors['tissue_hires_scalef']
    return scalefactor
    

def scale_tissue_positions(tissue_positions,
                           scalefactors,
                           image_scale):
    scale = 1
    if image_scale == 'lowres':
        scale = scalefactors['tissue_lowres_scalef']
    elif image_scale == 'hires':
        scale = scalefactors['tissue_hires_scalef']
    tissue_positions['x'] = tissue_positions['x'] * scale
    tissue_positions['y'] = tissue_positions['y'] * scale
    return tissue_positions


@dataclass
class Visium10X(BaseMolecularImaging):
    # TODO: Specify which Visium
    
    image: Image
    # Need a map from pos to ref_mat
    # TODO: Implement later. For now: If it works, it works!
    # pos_to_ref_map: dict
    table: Pointset
    scale_factors: dict
    __ref_mat: Annotation = field(init=False)
    spec_to_ref_map: dict = field(init=False)
    skip_ref_mat_creation: bool = False
    config: Optional[dict] = None
    spot_scaling_journal: Optional[PdDataframe] = None
    tissue_mask: Optional[numpy.array] = None
    background: ClassVar[int] = 0
    
    def __post_init__(self):
        if not self.skip_ref_mat_creation:
            self.__init_ref_mat()
        if self.spot_scaling_journal is None:
            self.__init_spot_scaling_journal()
            self.update_scaling_journal(operation_desc='init')
        self.id_ = uuid.uuid1()
    
    def __init_spot_scaling_journal(self):
        ref_idx = np.unique(self.__ref_mat.data)
        ref_idx = ref_idx[ref_idx!=self.background]
        self.spot_scaling_journal = pd.DataFrame(index=ref_idx)

    def __init_ref_mat(self):
        self.build_ref_mat()
        # self.__ref_mat = Annotation(self.build_ref_mat())

    @property
    def ref_mat(self):
        return self.__ref_mat
    
    @ref_mat.setter
    def ref_mat(self, ref_mat: Annotation):
        self.__ref_mat = ref_mat

    def store(self, root_directory: str):
        if not exists(root_directory):
            os.mkdir(root_directory)
        directory = join(root_directory, str(self.id_))
        if not exists(directory):
            os.mkdir(directory)
        f_dict = {}
        self.image.store(directory)
        f_dict['image'] = join(directory, str(self.image.id_))
        self.table.store(directory)
        f_dict['table'] = join(directory, str(self.table.id_))
        scale_factors_path = join(directory, 'scale_factors.json')
        with open(scale_factors_path, 'w') as f:
            json.dump(self.scale_factors, f)
        f_dict['scale_factors_path'] = scale_factors_path
        if self.__ref_mat is not None:
            self.__ref_mat.store(directory)
            f_dict['__ref_mat'] = join(directory, str(self.__ref_mat.id_))
        spec_to_ref_map_path = join(directory, 'spec_to_ref_mat.json')
        with open(spec_to_ref_map_path, 'w') as f:
            json.dump(self.spec_to_ref_map, f)
        f_dict['spec_to_ref_map_path'] = spec_to_ref_map_path
        if self.config is not None:
            config_path = join(directory, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(self.config, f)
            f_dict['config_path'] = config_path
        if self.spot_scaling_journal is not None:
            spot_scaling_journal_path = join(directory, 'spot_scaling_journal.csv')
            self.spot_scaling_journal.to_csv(spot_scaling_journal_path)
            f_dict['spot_scaling_journal_path'] = spot_scaling_journal_path
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
        spot_scaling_journal_path = attributes.get('spot_scaling_journal_path', None)
        if spot_scaling_journal_path is not None:
            spot_scaling_journal = pd.read_csv(spot_scaling_journal_path)
        else:
            spot_scaling_journal = None
        id_ = uuid.UUID(os.path.basename(directory.rstrip('/')))
        obj = cls(
            image=image,
            table=table,
            scale_factors=scale_factors,
            skip_ref_mat_creation=True,
            config=config,
            spot_scaling_journal=spot_scaling_journal
        )
        obj.ref_mat = __ref_mat
        obj.spec_to_ref_map = spec_to_ref_map
        obj.id_ = id_
        return obj
        

    def build_ref_mat(self):
        # TODO: Check whether "scale_ref_original_fullres" should be removed and replace with the original key.
        scalefactor = self.config['scalefactor']
        spot_diameter = int(round(self.scale_factors['spot_diameter_fullres'] * scalefactor))
        spot_radius = spot_diameter//2
        
        # Add numeric indices for the ref_mat
        self.spec_to_ref_map = {}
        # TODO: Clean up so that we remove int_idx information from table.
        ref_mat_idx_counter = 1
        for tbl_idx, _ in self.table.data.iterrows():
            self.spec_to_ref_map[tbl_idx] = ref_mat_idx_counter
            ref_mat_idx_counter += 1
        self.table.data['int_idx'] = range(1, self.table.data.shape[0] + 1)
        ref_mat = np.zeros((self.image.data.shape[0], self.image.data.shape[1]), dtype=np.int64)
        for tbl_idx, row in self.table.data.iterrows():
            x, y = row['x'], row['y']
            # int_idx = row['int_idx']
            int_idx = self.spec_to_ref_map[tbl_idx]
            xl = max(math.floor(x - spot_radius), 0)
            xh = min(math.ceil(x + spot_radius), self.image.data.shape[0])
            yl = max(math.floor(y - spot_radius), 0)
            yh = min(math.ceil(y + spot_radius), self.image.data.shape[1])

            for i in range(xl, xh):
                for j in range(yl, yh):
                    if np.sqrt((i - x)**2 + (j - y)**2) <= spot_radius:
                        ref_mat[i,j] = int_idx
        # TODO: See whether I can get ardount the np.float32. Can Greedy register with just ints.
        self.__ref_mat = Annotation(data=ref_mat.astype(np.float32))

    def update_scaling_journal(self, operation_desc=None):
        ref_idxs, counts = np.unique(self.__ref_mat.data, return_counts=True)
        new_col = pd.DataFrame(counts, index=ref_idxs, columns=[operation_desc])
        self.spot_scaling_journal = self.spot_scaling_journal.merge(new_col, left_index=True, right_index=True)

    @staticmethod
    def get_type() -> str:
        return 'Visium10X'
        
    def pad(self, padding: Tuple[int, int, int, int]):
        left, right, top, bottom = padding
        self.image.pad(padding)
        self.table.pad(padding)
        self.__ref_mat.pad(padding)
        self.update_scaling_journal(operation_desc=f'pad_data({padding})')

    def rescale(self, height: int, width: int):
        h_old, w_old = self.image.data.shape[0], self.image.data.shape[1]
        self.image.rescale(height, width)
        width_scale = width / w_old
        height_scale = height / h_old
        self.table.rescale(height_scale, width_scale)
        self.__ref_mat.rescale(height, width) 
        self.update_scaling_journal(operation_desc=f'rescale_data(height={height}, width={width}')

    def apply_bounding_parameters(self, x1: int, x2: int, y1: int, y2: int):
        self.image.apply_bounding_parameters(x1, x2, y1, y2)
        self.table.apply_bounding_parameters(x1, x2, y1, y2)
        # TODO: Should points that are out-of-bounds of the image be filtered as well?
        if self.__ref_mat is not None:
            self.__ref_mat.apply_bounding_parameters(x1, x2, y1, y2)
            self.update_scaling_journal(operation_desc=f'apply_bounding_parameters({x1}, {x2}, {y1}, {y2})')

    def flip(self, axis: int =0):
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
        obj = Visium10X(
            image=self.image.copy(),
            table=self.table.copy(),
            scale_factors=self.scale_factors.copy(),
            skip_ref_mat_creation=True,
            config=self.config.copy(),
            spot_scaling_journal=self.spot_scaling_journal.copy()
        )
        obj.spec_to_ref_map = self.spec_to_ref_map
        obj.ref_mat = ref_mat
        return obj

    def warp(self, 
             registerer: Registerer, 
             transformation: Any, 
             **kwargs: Dict) -> 'Visium10X':
        image_transformed = self.image.warp(registerer, transformation, **kwargs)
        ref_mat_warped = self.__ref_mat.warp(registerer, transformation, **kwargs)
        # TODO: See if we can get around the custom_max_voting_filter
        ref_mat_warped = Annotation(data=custom_max_voting_filter(ref_mat_warped.data))
        table = self.table.warp(registerer, transformation, **kwargs)
        config = self.config.copy() if self.config is not None else None
        transformed_st_data = Visium10X(
            image=image_transformed, 
            table=table, 
            scale_factors=self.scale_factors, 
            skip_ref_mat_creation=True,
            config=config,
            spot_scaling_journal=self.spot_scaling_journal.copy()
        )
        transformed_st_data.ref_mat = ref_mat_warped
        transformed_st_data.spec_to_ref_map = self.spec_to_ref_map.copy()
        transformed_st_data.update_scaling_journal(operation_desc='warping')
        return transformed_st_data

    def get_spec_to_ref_map(self, reverse=False):
        map_ = None
        if reverse:
            map_ = {self.spec_to_ref_map[x]: x for x in self.spec_to_ref_map}
        else:
            map_ = self.spec_to_ref_map.copy()
        return map_

    @classmethod
    def from_config(cls, config: Dict[str, str]) -> 'Visium10X':
        load_type = config['load_type']
        if load_type == 'spcrng_directory':
            spcrng_dir = config['spcrng_directory']
            image_scale = config['image_scale']
            if image_scale == 'fullres':
                image_path = config['path_to_fullres_image']
            else:
                image_path = None
            return Visium10X.from_spcrng(spcrng_dir, 
                                         image_scale, 
                                         image_path,
                                         config)

    @classmethod
    def from_spcrng(cls, 
                    directory: str,
                    image_scale: str ='hires',
                    fullres_image_path: str =None,
                    config: Dict =None):
        """
        Initiates Visium10X from spaceranger output directory.
        
        directory: output directory of spaceranger, typically named out.
        """
        # TODO: What should be the default file?
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
        return Visium10X.from_spcrng_files(path_to_scalefactors,
                                           path_to_tissue_positions,
                                           path_to_image,
                                           image_scale,
                                           config)
    
    @classmethod
    def from_spcrng_files(cls,
                          path_to_scalefactors: str,
                          path_to_tissue_positions: str,
                          path_to_image: str,
                          image_scale: str ='hires',
                          config: Dict =None):
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
        tissue_positions = Pointset(data=tissue_positions_df)
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