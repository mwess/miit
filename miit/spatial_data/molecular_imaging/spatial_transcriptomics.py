from dataclasses import dataclass
import json
import math
import os
from os.path import join, exists
from typing import Any, ClassVar, Dict, Optional, Tuple
import uuid

import cv2
import SimpleITK as sitk
import numpy as np
import pandas
import pandas as pd

from miit.spatial_data.molecular_imaging.imaging_data import BaseMolecularImaging
from miit.registerers.base_registerer import Registerer
from miit.utils.utils import custom_max_voting_filter
from miit.spatial_data.image import Annotation, Image, Pointset

@dataclass
class SpatialTranscriptomics(BaseMolecularImaging):
    
    image: Image
    table: Pointset
    scale_factors: dict
    skip_index_mask_creation: bool = False
    index_mask: Optional[Annotation] = None
    config: Optional[dict] = None
    spot_scaling_journal: Optional[pandas.DataFrame] = None
    background: ClassVar[int] = 0
    
    def __post_init__(self):
        if not self.skip_index_mask_creation:
            self.__init_index_mask()
        if self.spot_scaling_journal is None:
            self.__init_spot_scaling_journal()
            self.update_scaling_journal(operation_desc='init')
        self.id_ = uuid.uuid1()
    
    def __init_spot_scaling_journal(self):
        ref_idx = np.unique(self.index_mask.data)
        ref_idx = ref_idx[ref_idx!=self.background]
        self.spot_scaling_journal = pd.DataFrame(index=ref_idx)

    def __init_index_mask(self):
        self.index_mask = Annotation(self.build_index_mask())

    def store(self, directory):
        if not exists(directory):
            os.mkdir(directory)
        sub_dir = join(directory, str(self.id_))
        if not exists(sub_dir):
            os.mkdir(sub_dir)
        sitk.WriteImage(sitk.GetImageFromArray(self.image.data), join(sub_dir, 'image.nii.gz'))
        self.table.data.to_csv(join(sub_dir, 'table.csv'))
        with open(join(sub_dir, 'scale_factors.json'), 'w') as f:
            json.dump(self.scale_factors, f)
        if self.index_mask is not None:
            sitk.WriteImage(sitk.GetImageFromArray(self.index_mask.data), join(sub_dir, 'index_mask.nii.gz'))
        if self.config is not None:
            with open(join(sub_dir, 'config.json'), 'w') as f:
                json.dump(self.config, f)
        if self.spot_scaling_journal is not None:
            self.spot_scaling_journal.to_csv(join(sub_dir, 'spot_scaling_journal.csv'))

    def build_index_mask(self):
        spot_diameter = int(round(self.scale_factors['spot_diameter_fullres'] * self.scale_factors['tissue_original_scalef']))
        spot_radius = spot_diameter//2
        
        # Add numeric indices for the index_mask
        self.table.data['int_idx'] = range(1, self.table.data.shape[0] + 1)
        index_mask = np.zeros((self.image.data.shape[0], self.image.data.shape[1]), dtype=np.int64)
        for _, row in self.table.data.iterrows():
            x, y = row['x'], row['y']
            int_idx = row['int_idx']
            xl = max(math.floor(x - spot_radius), 0)
            xh = min(math.ceil(x + spot_radius), self.image.data.shape[0])
            yl = max(math.floor(y - spot_radius), 0)
            yh = min(math.ceil(y + spot_radius), self.image.data.shape[1])

            for i in range(xl, xh):
                for j in range(yl, yh):
                    if np.sqrt((i - x)**2 + (j - y)**2) <= spot_radius:
                        index_mask[i,j] = int_idx
        return index_mask.astype(np.float32)

    def update_scaling_journal(self, operation_desc=None):
        ref_idxs, counts = np.unique(self.index_mask.data, return_counts=True)
        new_col = pd.DataFrame(counts, index=ref_idxs, columns=[operation_desc])
        self.spot_scaling_journal = self.spot_scaling_journal.merge(new_col, left_index=True, right_index=True)

    @staticmethod
    def get_type() -> str:
        return 'SpatialTranscriptomics'
        
    def pad(self, padding: Tuple[int, int, int, int]):
        left, right, top, bottom = padding
        self.image.pad(padding)
        self.table.pad(padding)
        self.index_mask.pad(padding)
        self.update_scaling_journal(operation_desc=f'pad_data({padding})')

    def rescale(self, height: int, width: int):
        h_old, w_old = self.image.data.shape[0], self.image.data.shape[1]
        self.image.rescale(height, width)
        width_scale = width / w_old
        height_scale = height / h_old
        self.table.rescale(height_scale, width_scale)
        self.index_mask.rescale(height, width) 
        self.update_scaling_journal(operation_desc=f'rescale_data(height={height}, width={width}')

    def apply_bounding_parameters(self, x1: int, x2: int, y1: int, y2: int):
        self.image.apply_bounding_parameters(x1, x2, y1, y2)
        self.table.apply_bounding_parameters(x1, x2, y1, y2)
        # TODO: Should points that are out-of-bounds of the image be filtered as well?
        if self.index_mask is not None:
            self.index_mask.apply_bounding_parameters(x1, x2, y1, y2)
            self.update_scaling_journal(operation_desc=f'apply_bounding_parameters({x1}, {x2}, {y1}, {y2})')

    def flip(self, axis: int = 0):
        pass

    def copy(self):
        if self.index_mask is not None:
            index_mask = self.index_mask.copy()
        else:
            index_mask = None
        return SpatialTranscriptomics(
            image=self.image.copy(),
            table=self.table.copy(),
            scale_factors=self.scale_factors.copy(),
            skip_index_mask_creation=True,
            index_mask=index_mask,
            config=self.config.copy(),
            spot_scaling_journal=self.spot_scaling_journal.copy()
        )

    def warp(self, registerer: Registerer, transformation: Any, args: Optional[Dict[Any, Any]] = None) -> 'SpatialTranscriptomics':
        if args is None:
            args = {}
        image_transformed = self.image.warp(registerer, transformation, args)
        index_mask_warped = self.index_mask.warp(registerer, transformation, args)
        index_mask_warped = Annotation(custom_max_voting_filter(index_mask_warped.data))
        table = self.table.warp(registerer, transformation, args)
        config = self.config.copy() if self.config is not None else None
        transformed_st_data = SpatialTranscriptomics(
            image=image_transformed, 
            table=table, 
            scale_factors=self.scale_factors, 
            skip_index_mask_creation=True,
            index_mask=index_mask_warped,
            config=config,
            spot_scaling_journal=self.spot_scaling_journal.copy()
        )
        transformed_st_data.update_scaling_journal(operation_desc='warping')
        return transformed_st_data

    @classmethod
    def from_basedir(cls, path: str, key: str):
        # Path points to the basedir
        image_path = os.path.join(path, 'original_images', key + '.tif')
        image = Image(cv2.imread(image_path))
        
        table_path = os.path.join(path, 'st_data', key + '.csv')
        table = Pointset(pd.read_csv(table_path, index_col=0))
        
        scale_factors_path = os.path.join(path, 'st_scalefactors', key + '.json')
        with open(scale_factors_path) as f:
            scale_factors = json.load(f)
        return cls(image, table, scale_factors)

    @classmethod
    def from_config(cls, config: Dict[str, str]) -> 'SpatialTranscriptomics':
        image_path = config['image']
        table_path = config['st_data']
        scale_factors_path = config['st_scalefactors']
        
        image = Image(cv2.imread(image_path))
        table = pd.read_csv(table_path, index_col=0)
        table = table[['imagerow', 'imagecol']].rename(columns={'imagerow': 'x', 'imagecol': 'y'})
        table = Pointset(table)
        with open(scale_factors_path) as f:
            scale_factors = json.load(f)
        return cls(image, table, scale_factors, config=config)

    @classmethod
    def from_directory(cls, directory: str) -> 'SpatialTranscriptomics':
        """
        Loads st object from supplied directory.
        """
        if not exists(directory):
            # TODO: Throw error
            pass
        image = Image(sitk.GetArrayFromImage(sitk.ReadImage(join(directory, 'image.nii.gz'))))
        table = Pointset(pd.read_csv(join(directory, 'table.csv'), index_col=0))
        with open(join(directory, 'scale_factors.json')) as f:
            scale_factors = json.load(f)
        config_path = join(directory, 'config.json')
        if exists(config_path): 
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = None
        ref_mat_path = join(directory, 'index_mask.nii.gz')
        if exists(ref_mat_path):
            index_mask = Annotation(sitk.GetArrayFromImage(sitk.ReadImage(ref_mat_path)))
            skip_index_mask_creation = True
        else:
            index_mask = None
            skip_index_mask_creation = False
        spot_scaling_journal_path = join(directory, 'spot_scaling_journal.csv')
        if exists(spot_scaling_journal_path):
            spot_scaling_journal = pd.read_csv(spot_scaling_journal_path, index_col=0)
        else:
            spot_scaling_journal = None
        # strip directory somehow
        id_ = uuid.UUID(os.path.basename(directory))
        # TODO: Parse id_ into cls object
        return cls(
            image=image,
            table=table,
            scale_factors=scale_factors,
            skip_index_mask_creation=skip_index_mask_creation,
            index_mask=index_mask,
            config=config,
            spot_scaling_journal=spot_scaling_journal
        )
        
         