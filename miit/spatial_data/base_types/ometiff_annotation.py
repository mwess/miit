from dataclasses import dataclass
import json
from os.path import join, exists
from typing import Dict, List, Optional, Union
import uuid


import numpy, numpy as np
import tifffile


# from .ometiff_image import OMETIFFImage
from .annotation import Annotation
from miit.utils.utils import create_if_not_exists
from miit.utils.tif_utils import get_tif_metadata, write_ome_tif_file, write_tif_file


@dataclass(kw_only=True)
class OMETIFFAnnotation(Annotation):
    
    # interpolation_mode: ClassVar[str] = 'NN'
    # labels: Optional[Union[List[str], Dict[str, int]]] = None
    keep_axis_orientation: bool = False
    is_ome: bool = True
    tif_metadata: Optional[Dict] = None
    
    def store(self, path: str):
        create_if_not_exists(path)
        image_fname = 'image.ome.tif' if self.is_ome else 'image.tif'
        image_path = join(path, image_fname)
        self.__to_file(image_path)
        additional_attributes = {
            'id': str(self._id),
            'name': self.name,
            'meta_information': self.meta_information,
            'is_ome': self.is_ome,
            'keep_axis_orientation': self.keep_axis_orientation,
            'tif_metadata': self.tif_metadata,
            'is_multichannel': self.is_multichannel
        }
        with open(join(path, 'additional_attributes.json'), 'w') as f:
            json.dump(additional_attributes, f)
        if self.labels:
            with open(join(path, 'labels.json'), 'w') as f:
                json.dump(self.labels, f)

    @staticmethod
    def get_type() -> str:
        return 'ometiff_annotation'
    
    def get_resolution(self):
        return float(self.tif_metadata['PhysicalSizeX'])
            
    def __to_file(self, path: str):
        if not self.keep_axis_orientation and len(self.data.shape) > 2:
            data = np.moveaxis(self.data, 2, 0)
        else:
            data = self.data
        if self.is_ome:
            write_ome_tif_file(path, data, self.tif_metadata)
        else:
            write_tif_file(path, data)
    
    @classmethod
    def load(cls, path: str):
        aa_path = join(path, 'additional_attributes.json') 
        with open(aa_path) as f:
            additional_attributes = json.load(f)
        is_ome = additional_attributes['is_ome']
        is_multichannel = additional_attributes['is_multichannel']
        image_fname = 'image.ome.tif' if is_ome else 'image.tif'
        image_path = join(path, image_fname)           
        keep_axis_orientation = additional_attributes['keep_axis_orientation']
        tif = tifffile.TiffFile(image_path)
        tif_metadata = get_tif_metadata(tif)
        name = additional_attributes['name']
        meta_information = additional_attributes['meta_information']
        _id = uuid.UUID(additional_attributes['id'])
        labels_path = join(path, 'labels.json')
        if exists(labels_path):
            with open(labels_path) as f:
                labels = json.load(f)
        data = tif.asarray()
        if not keep_axis_orientation and len(data.shape) > 2:
            data = np.moveaxis(data, 0, 2)
        ometiff_image = cls(
            data=data,
            name=name,
            meta_information=meta_information,
            is_ome=is_ome,
            labels=labels,
            keep_axis_orientation=keep_axis_orientation,
            tif_metadata=tif_metadata,
            is_multichannel=is_multichannel
        )
        ometiff_image._id=_id
        return ometiff_image

    @classmethod
    def load_from_path(cls, 
                       path: str, 
                       keep_axis_orientation=False, 
                       is_multichannel=False,
                       labels: Optional[Union[List, Dict]] = None,
                       name: str = ''):
        tif = tifffile.TiffFile(path)
        data = tif.asarray()
        if not keep_axis_orientation and len(data.shape) > 2:
            data = np.moveaxis(data, 0, 2)
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
                   labels=labels,
                   keep_axis_orientation=keep_axis_orientation,
                   tif_metadata=tif_metadata,
                   is_multichannel=is_multichannel)