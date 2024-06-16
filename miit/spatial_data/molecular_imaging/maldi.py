from dataclasses import dataclass
import json
import os
from os.path import join, exists
from typing import Any, ClassVar, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas
import pandas as pd
import SimpleITK as sitk

from miit.spatial_data.molecular_imaging.imaging_data import BaseMolecularImaging
from miit.spatial_data.image import Annotation, Image
from miit.registerers.base_registerer import Registerer
from miit.utils.utils import custom_max_voting_filter, copy_if_not_none


@dataclass
class Maldi(BaseMolecularImaging):

    #TODO: Rewrite this to include new MALDI format.
    
    image: Image 
    skip_index_mask_creation: bool = False
    index_mask: Optional[Annotation] = None
    imzml_path: Optional[str] = None
    ibd_path: Optional[str] = None
    config: Optional[dict] = None
    spot_scaling_journal: Optional[pandas.DataFrame] = None
    background: ClassVar[int] = -1

    def __post_init__(self):
        if self.spot_scaling_journal is None:
            self.__init_spot_scaling_journal()
            self.update_scaling_journal(operation_desc='init')

    def __init_spot_scaling_journal(self):
        ref_idx = np.unique(self.index_mask.data)
        ref_idx = ref_idx[ref_idx != self.background]
        self.spot_scaling_journal = pd.DataFrame(index=ref_idx)
        
    def update_scaling_journal(self, operation_desc=None):
        ref_idxs, counts = np.unique(self.index_mask.data, return_counts=True)
        new_col = pd.DataFrame(counts, index=ref_idxs, columns=[operation_desc])
        self.spot_scaling_journal = self.spot_scaling_journal.merge(new_col, left_index=True, right_index=True)

    @staticmethod
    def get_type() -> str:
        return 'Maldi'

    def pad(self, padding: Tuple[int, int, int, int]):
        self.image.pad(padding)
        self.index_mask.pad(padding, constant_values=self.background)
        operation_desc = f'pad_data({padding})'
        self.update_scaling_journal(operation_desc)

    def rescale(self, height: int, width: int):
        self.image.rescale(height, width)
        self.index_mask.rescale(height, width)
        operation_desc = f'rescale_data(height={height}, width={width})'
        self.update_scaling_journal(operation_desc=operation_desc)

    def apply_bounding_parameters(self, x1: int, x2: int, y1: int, y2: int):
        self.image.apply_bounding_parameters(x1, x2, y1, y2)
        self.index_mask.apply_bounding_parameters(x1, x2, y1, y2)
        operation_desc = f'apply_bounding_box({x1}, {x2}, {y1}, {y2})'
        self.update_scaling_journal(operation_desc=operation_desc)

    def flip(self, axis: int =0):
        pass

    def copy(self):
        index_mask = copy_if_not_none(self.index_mask)
        return Maldi(
            image=self.image.copy(),
            skip_index_mask_creation=True,
            index_mask=index_mask,
            imzml_path=self.imzml_path,
            ibd_path=self.ibd_path,
            config=self.config.copy(),
            spot_scaling_journal=self.spot_scaling_journal.copy()
        )

    def warp(self, registerer: Registerer, transformation: Any, args: Optional[Dict[Any, Any]] = None) -> 'Maldi':
        if args is None:
            args = {}
        args = args.copy()
        image_transformed = self.image.warp(registerer, transformation, args)
        # Copy index mask so that we can manipulate indices without having to keep track of the wrt. to original data.
        index_mask = self.index_mask.copy()
        index_mask.data += 1
        index_mask_transformed = index_mask.warp(registerer, transformation, args)
        index_mask_transformed = Annotation(custom_max_voting_filter(index_mask_transformed.data))
        index_mask_transformed.data = index_mask_transformed.data - 1
        config = self.config.copy() if self.config is not None else None
        maldi_transformed = Maldi(
            image=image_transformed, 
            index_mask=index_mask_transformed,
            imzml_path=self.imzml_path,
            ibd_path=self.ibd_path,
            skip_index_mask_creation=True,
            config=config,
            spot_scaling_journal=self.spot_scaling_journal.copy())
        maldi_transformed.update_scaling_journal(operation_desc='warping')
        return maldi_transformed

    def store(self, directory: str):
        if not exists(directory):
            os.mkdir(directory)
        self.image.store(directory)
        if self.index_mask is not None:
            self.index_mask.store(directory, fname='index_mask.nii.gz')
        additional_params = {}
        if self.imzml_path is not None:
            additional_params['imzml_path'] = self.imzml_path
        if self.ibd_path is not None:
            additional_params['ibd_path'] = self.ibd_path
        with open(join(directory, 'attributes.json'), 'w') as f:
            json.dump(additional_params, f)
        if self.config is not None:
            with open(join(directory, 'config.json'), 'w') as f:
                json.dump(self.config, f)
        if self.spot_scaling_journal is not None:
            self.spot_scaling_journal.to_csv(join(directory, 'spot_scaling_journal.csv'))
    
    @classmethod
    def from_basedir(cls, path, key):
        pass

    @classmethod
    def from_config(cls, config):
        image_path = config['image']
        image = Image(cv2.imread(image_path))
        index_mask_path = config['reference_matrix']
        index_mask = Annotation(sitk.GetArrayFromImage(sitk.ReadImage(index_mask_path)))
        imzml_path = config['imzml']
        ibd_path = config['ibd']
        return cls(
            image=image,
            skip_index_mask_creation=True,
            index_mask=index_mask,
            imzml_path=imzml_path,
            ibd_path=ibd_path,
            config=config
        )

    @classmethod
    def from_directory(cls, directory: str) -> 'Maldi':
        if not exists(directory):
            # Throw error
            pass
        image = Image(sitk.GetArrayFromImage(sitk.ReadImage(join(directory, 'image.nii.gz'))))
        ref_mat_path = join(directory, 'index_mask.nii.gz')
        if exists(ref_mat_path):
            index_mask = Annotation(sitk.GetArrayFromImage(sitk.ReadImage(ref_mat_path)))
            skip_index_mask_creation = True
        else:
            index_mask = False
            skip_index_mask_creation = False
        with open(join(directory, 'attributes.json')) as f:
            attributes = json.load(f)
        imzml_path = attributes.get('imzml_path', None)
        ibd_path = attributes.get('ibd_path', None)
        config_path = join(directory, 'config.json')
        if exists(config_path): 
            with open(config_path) as f:
                config = json.load(f)
        else:
            config = None
        spot_scaling_journal_path = join(directory, 'spot_scaling_journal.csv')
        if exists(spot_scaling_journal_path):
            spot_scaling_journal = pd.read_csv(spot_scaling_journal_path, index_col=0)
        else:
            spot_scaling_journal = None
        return cls(
            image=image,
            skip_index_mask_creation=skip_index_mask_creation,
            index_mask=index_mask,
            imzml_path=imzml_path,
            ibd_path=ibd_path,
            config=config,
            spot_scaling_journal=spot_scaling_journal
        )
