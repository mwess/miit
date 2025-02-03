from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
from os.path import join
from typing import Any
import uuid

import numpy

from greedyfhist.utils.io import read_image, write_to_ometiffile
from .image import Image
from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.utils import create_if_not_exists
from miit.utils.distance_unit import DUnit


def get_default_metadata() -> dict:
    metadata = {
        'PhysicalSizeX': 1,
        'PhysicalSizeXUnit': 'px',
        'PhysicalSizeY': 1,
        'PhysicalSizeYUnit': 'px',
        'Interleaved': 'true',
        'channels': [],
        'tiff_data': []
    }
    return metadata


# TODO: Add resolution to file parsing.
@dataclass(kw_only=True)
class OMETIFFImage(Image):
    """
    Class for processing TIFF and OMETIFF images. 
    
    
    Attributes
    ----------
    
    data: numpy.ndarray
        Image data as a numpy array. 
        
    path: str
        Original file path.
        
    tif: tifffile.tifffile.TiffFile 
        Connection to tif image information.

    is_ome: bool
        Indicates whether the read file is ome.
    """
    # interpolation_mode: ClassVar[str] = 'LINEAR'
    is_ome: bool = True
    tif_metadata: dict = field(default_factory=lambda: defaultdict(dict))
    
    def __post_init__(self):
        super().__post_init__()
        if self.tif_metadata is not None:
            x_size = self.tif_metadata.get('PhysicalSizeX', 1.0)
            x_unit = self.tif_metadata.get('PhysicalSizeXUnit', 'px')
            y_size = self.tif_metadata.get('PhysicalSizeY', 1.0)
            y_unit = self.tif_metadata.get('PhysicalSizeYUnit', 'px')
            self.resolution = (DUnit(x_size, x_unit), DUnit(y_size, y_unit))
            
    
    def resize(self, width: int, height: int):
        # Use opencv's resize function here, because it typically works a lot faster and for now
        # we assume that data in Image is always some kind of rgb like image.
        w, h = self.data.shape[:2]
        w_spacing = self.tif_metadata['PhysicalSizeX']
        h_spacing = self.tif_metadata['PhysicalSizeY']
        w_spacing_new = w_spacing * w / width
        h_spacing_new = h_spacing * h / height
        self.tif_metadata['PhysicalSizeX'] = w_spacing_new
        self.tif_metadata['PhysicalSizeY'] = h_spacing_new
        super().resize(width, height)

    def get_resolution(self):
        return float(self.tif_metadata['PhysicalSizeX'])

    def get_spacing(self) -> tuple[float, float]:
        w_spacing = self.tif_metadata.get('PhysicalSizeX', 1)
        h_spacing = self.tif_metadata.get('PhysicalSizeY', 1)
        return (w_spacing, h_spacing)

    def apply_transform(self, 
                        registerer: Registerer, 
                        transformation: RegistrationResult | numpy.ndarray, 
                        **kwargs: dict) -> Any:
        transformed_image = self.transform(registerer, transformation, **kwargs)
        return OMETIFFImage(
            data=transformed_image,
            name=self.name,
            meta_information=self.meta_information.copy(),
            is_ome=self.is_ome,
            tif_metadata=self.tif_metadata.copy()
        )

    def store(self, path: str):
        create_if_not_exists(path)
        image_fname = 'image.ome.tif'
        image_path = join(path, image_fname)
        self.to_file__(image_path)
        additional_attributes = {
            'name': self.name,
            'meta_information': self.meta_information,
            'is_ome': self.is_ome,
            'id': str(self._id)
        }
        with open(join(path, 'additional_attributes.json'), 'w') as f:
            json.dump(additional_attributes, f)

    def to_file__(self, path: str):
        write_to_ometiffile(
            self.data, path, self.tif_metadata, False
        )

    @staticmethod
    def get_type() -> str:
        return 'ometiff_image'       

    @classmethod
    def load(cls, path: str) -> 'OMETIFFImage':
        aa_path = join(path, 'additional_attributes.json') 
        with open(aa_path) as f:
            additional_attributes = json.load(f)
        is_ome = additional_attributes['is_ome']
        image_fname = 'image.ome.tif'
        image_path = join(path, image_fname)           
        data, metadata = read_image(image_path)
        name = additional_attributes['name']
        meta_information = additional_attributes['meta_information']
        _id = uuid.UUID(additional_attributes['id'])
        ometiff_image = cls(
            data=data,
            name=name,
            meta_information=meta_information,
            is_ome=is_ome,
            tif_metadata=metadata
        )
        ometiff_image._id=_id
        return ometiff_image

    @classmethod
    def load_from_path(cls, 
                       path: str,
                       name: str = '') -> 'OMETIFFImage':
        data, metadata = read_image(path, False)
        if not metadata:
            metadata = get_default_metadata()
        if not name:
            name = os.path.basename(path)
        return cls(data=data, name=name, tif_metadata=metadata)