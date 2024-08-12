from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
from os.path import join
from typing import Any, Dict, Optional, Tuple
import uuid
import xml.etree.ElementTree as ET


import cv2
import numpy, numpy as np
import tifffile


from .base_imaging import BaseImage
from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.utils import create_if_not_exists


def derive_output_path(directory: str, fname: str, limit: int = 1000) -> str:
    """Generates a unique output path. If path is already existing,
    adds a counter value until a unique path is found.

    Args:
        directory (str): target directory
        fname (str): target filename
        limit (int, optional): Limit number to prevent endless loops. Defaults to 1000.

    Returns:
        str: Target path
    """
    target_path = join(directory, fname)
    if not os.path.exists(target_path):
        return target_path
    for suffix in range(limit):
        new_target_path = f'{target_path}_{suffix}'
        if not os.path.exists(new_target_path):
            return new_target_path
    return target_path


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

    __custom_interpolation_mode: str = 'LINEAR'
    is_ome: bool
    keep_axis_orientation: bool = False
    tif_metadata: Dict = field(default_factory=defaultdict(dict))
    __is_annotation: bool = False

    def __post_init__(self):
        self._id = uuid.uuid1()
        if self.__is_annotation:
            self.__custom_interpolation_mode = 'NN'

    @property
    def is_annotation(self):
        return self.__is_annotation
    
    @is_annotation.setter
    def is_annotation(self, is_annotation):
        self.__is_annotation = is_annotation
        if self.__is_annotation:
            self.__custom_interpolation_mode = 'NN'
        else:
            self.__custom_interpolation_mode = 'LINEAR'
    
    def transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
        image_transformed = registerer.transform_image(self.data, transformation, self.__custom_interpolation_mode, **kwargs)
        return image_transformed

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
            keep_axis_orientation=self.keep_axis_orientation,
            __is_annotation=self.__is_annotation,
            __tif_metadata=self.tif_metadata
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
            'keep_axis_orientation': self.keep_axis_orientation,
            '__is_annotation': self.__is_annotation,
            'tif_metadata': self.tif_metadata,
            'id': str(self._id)
        }
        with open(join(path, 'additional_attributes.json'), 'w') as f:
            json.dump(additional_attributes, f)

    def __to_file(self, path: str):
        if self.is_ome:
            self.__to_ome_tiff_file(path)
        else:
            self.__to_tiff_file(path)

    def __to_tiff_file(self, path: str):
        if self.is_annotation and not self.keep_axis_orientation and len(self.data.shape) > 2:
            data = np.moveaxis(self.data, 2, 0)
        else:
            data = self.data
        tifffile.imwrite(path, data)        

    def __to_ome_tiff_file(self, path: str, options: Optional[Dict] = None):
        if options is None:
            if len(self.data.shape) == 2:
                options = {}
            else:
                options = dict(photometric='rgb')
        if self.is_annotation and not self.keep_axis_orientation and len(self.data.shape) > 2:
            data = np.moveaxis(self.data, 2, 0)
        else:
            data = self.data

        with tifffile.TiffWriter(path, bigtiff=True) as tif:
            tif.write(
                data,
                metadata=self.tif_metadata,
                **options
            )

    @staticmethod
    def get_type() -> str:
        return 'ometiff_image'    

    @staticmethod
    def __get_metadata(tif: tifffile.tifffile.TiffFile) -> Dict:
        root = ET.fromstring(tif.ome_metadata)
        ns = {'ns0': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
        elem = root.findall('*/ns0:Pixels', ns)[0]
        metadata = elem.attrib
        # metadata['axes'] = 'TCYXS'
        del metadata['SizeX']
        del metadata['SizeY']
        del metadata['SizeC']
        del metadata['SizeT']
        del metadata['SizeZ']
        del metadata['DimensionOrder']
        return metadata       

    @classmethod
    def load(cls, path: str):
        aa_path = join(path, 'additional_attributes.json') 
        with open(aa_path) as f:
            additional_attributes = json.load(f)
        is_ome = additional_attributes['is_ome']
        image_fname = 'image.ome.tif' if is_ome else 'image.tif'
        image_path = join(path, image_fname)           
        keep_axis_orientation = additional_attributes['keep_axis_orientation']
        __is_annotation = additional_attributes['__is_annotation']
        tif = tifffile.TiffFile(image_path)
        tif_metadata = OMETIFFImage.__get_metadata(tif)
        name = additional_attributes['name']
        meta_information = additional_attributes['meta_information']
        _id = additional_attributes['id']
     
        data = tif.asarray()
        if __is_annotation and not keep_axis_orientation and len(data.shape) > 2:
            data = np.moveaxis(data, 0, 2)
        ometiff_image = cls(
            data=data,
            name=name,
            meta_information=meta_information,
            is_ome=is_ome,
            keep_axis_orientation=keep_axis_orientation,
            tif_metadata=tif_metadata
        )
        ometiff_image._id=_id
        return ometiff_image

    @classmethod
    def load_from_path(cls, 
                       path: str, 
                       keep_axis_orientation=False, 
                       is_annotation=False,
                       name: str = ''):
        tif = tifffile.TiffFile(path)
        data = tif.asarray()
        if is_annotation and not keep_axis_orientation and len(data.shape) > 2:
            data = np.moveaxis(data, 0, 2)
        if path.endswith('ome.tif') or path.endswith('ome.tiff'):
            is_ome = True
        else:
            is_ome = False
        meta_information = {
            'path': path
        }
        tif_metadata = OMETIFFImage.__get_metadata(tif)
        return cls(data=data,
                   name=name, 
                   meta_information=meta_information,
                   is_ome=is_ome,
                   keep_axis_orientation=keep_axis_orientation,
                   tif_metadata=tif_metadata)