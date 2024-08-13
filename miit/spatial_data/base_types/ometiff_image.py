from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
from os.path import join
from typing import Any, ClassVar, Dict, Optional, Tuple
import uuid
import xml.etree.ElementTree as ET


import cv2
import numpy, numpy as np
import tifffile


from .base_imaging import BaseImage
from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.utils import create_if_not_exists
from miit.utils.tif_utils import get_tif_metadata, write_tif_file, write_ome_tif_file




@dataclass(kw_only=True)
class OMETIFFImage(BaseImage):
    """
    Class for processing TIFF and OMETIFF images. 
    
    
    Attributes
    ----------
    
    data: numpy.array
        Image data as a numpy array. 
        
    path: str
        Original file path.
        
    tif: tifffile.tifffile.TiffFile 
        Connection to tif image information.

    is_ome: bool
        Indicates whether the read file is ome.
        
    is_annotation: bool
        If OME TIFF is an annotation this should be set to True, so 
        that Nearest Neighbor interpolation is used during 
        transformation.
        
    keep_axis: bool
        If set to True, first and third channel are not switched if 
        is_annotation is also true. Normally, images are provided 
        in W x H x C format, but some tools generate annotations in 
        format C x H x W, e.g. multi class annotations from QuPath. 
    """
    interpolation_mode: ClassVar[str] = 'LINEAR'
    is_ome: bool = True
    tif_metadata: Dict = field(default_factory=defaultdict(dict))

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.data = self.data[xmin:xmax, ymin:ymax]

    def resize(self, width, height):
        # Use opencv's resize function here, because it typically works a lot faster and for now
        # we assume that data in Image is always some kind of rgb like image.
        self.data = cv2.resize(self.data, (height, width))

    def pad(self, padding: Tuple[int, int, int, int], constant_values: int = 0):
        left, right, top, bottom = padding
        self.data = cv2.copyMakeBorder(self.data, top, bottom, left, right, cv2.BORDER_CONSTANT, constant_values)

    def flip(self, axis: int = 0):
        self.data = np.flip(self.data, axis=axis)    

    def get_resolution(self):
        return float(self.tif_metadata['PhysicalSizeX'])

    def apply_transform(self, registerer: Registerer, transformation: Any, **kwargs: Dict) -> Any:
        transformed_image = self.transform(registerer, transformation, **kwargs)
        return OMETIFFImage(
            data=transformed_image,
            name=self.name,
            meta_information=self.meta_information.copy(),
            is_ome=self.is_ome,
            tif_metadata=self.tif_metadata
        )

    def store(self, path: str):
        create_if_not_exists(path)
        image_fname = 'image.ome.tif' if self.is_ome else 'image.tif'
        image_path = join(path, image_fname)
        self.__to_file(image_path)
        additional_attributes = {
            'name': self.name,
            'meta_information': self.meta_information,
            'is_ome': self.is_ome,
            'tif_metadata': self.tif_metadata,
            'id': str(self._id)
        }
        with open(join(path, 'additional_attributes.json'), 'w') as f:
            json.dump(additional_attributes, f)

    def __to_file(self, path: str):
        if self.is_ome:
            write_ome_tif_file(
                path,
                self.data,
                self.tif_metadata
            )
        else:
            write_tif_file(path, self.data)

    @staticmethod
    def get_type() -> str:
        return 'ometiff_image'       

    @classmethod
    def load(cls, path: str):
        aa_path = join(path, 'additional_attributes.json') 
        with open(aa_path) as f:
            additional_attributes = json.load(f)
        is_ome = additional_attributes['is_ome']
        image_fname = 'image.ome.tif' if is_ome else 'image.tif'
        image_path = join(path, image_fname)           
        tif = tifffile.TiffFile(image_path)
        tif_metadata = get_tif_metadata(tif)
        name = additional_attributes['name']
        meta_information = additional_attributes['meta_information']
        _id = uuid.UUID(additional_attributes['id'])
     
        data = tif.asarray()
        ometiff_image = cls(
            data=data,
            name=name,
            meta_information=meta_information,
            is_ome=is_ome,
            tif_metadata=tif_metadata
        )
        ometiff_image._id=_id
        return ometiff_image

    @classmethod
    def load_from_path(cls, 
                       path: str, 
                       name: str = ''):
        tif = tifffile.TiffFile(path)
        data = tif.asarray()
        if path.endswith('ome.tif') or path.endswith('ome.tiff'):
            is_ome = True
        else:
            is_ome = False
        meta_information = {
            'path': path
        }
        tif_metadata = get_tif_metadata(tif)
        return cls(data=data,
                   name=name, 
                   meta_information=meta_information,
                   is_ome=is_ome,
                   tif_metadata=tif_metadata)