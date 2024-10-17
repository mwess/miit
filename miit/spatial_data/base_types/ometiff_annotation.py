from dataclasses import dataclass
import json
from os.path import join, exists
from typing import Dict, List, Optional, Union, Tuple
import uuid

import numpy as np

from .annotation import Annotation
from .ometiff_image import get_default_metadata
from greedyfhist.utils.io import read_image, write_to_ometiffile
from miit.utils.utils import create_if_not_exists


@dataclass(kw_only=True)
class OMETIFFAnnotation(Annotation):
    
    # interpolation_mode: ClassVar[str] = 'NN'
    # labels: Optional[Union[List[str], Dict[str, int]]] = None
    keep_axis_orientation: bool = False
    is_ome: bool = True
    tif_metadata: Optional[Dict] = None
    
    def store(self, path: str):
        create_if_not_exists(path)
        image_fname = 'image.ome.tif'
        image_path = join(path, image_fname)
        self.__to_file(image_path)
        additional_attributes = {
            'id': str(self._id),
            'name': self.name,
            'meta_information': self.meta_information,
            'is_ome': self.is_ome,
            'keep_axis_orientation': self.keep_axis_orientation,
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

    def resize(self, width: float, height: float):
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
    
    def get_spacing(self) -> Tuple[float, float]:
        w_spacing = self.tif_metadata.get('PhysicalSizeX', 1)
        h_spacing = self.tif_metadata.get('PhysicalSizeY', 1)
        return (w_spacing, h_spacing)

    def get_resolution(self):
        return float(self.tif_metadata['PhysicalSizeX'])
            
    def __to_file(self, path: str):
        if not self.keep_axis_orientation and len(self.data.shape) > 2:
            data = np.moveaxis(self.data, 2, 0)
        else:
            data = self.data
        write_to_ometiffile(
            data, path, self.tif_metadata, True
        )
    
    @classmethod
    def load(cls, path: str):
        aa_path = join(path, 'additional_attributes.json') 
        with open(aa_path) as f:
            additional_attributes = json.load(f)
        is_ome = additional_attributes['is_ome']
        is_multichannel = additional_attributes['is_multichannel']
        image_fname = 'image.ome.tif'
        image_path = join(path, image_fname)           
        keep_axis_orientation = additional_attributes['keep_axis_orientation']
        data, tif_metadata = read_image(image_path)
        name = additional_attributes['name']
        meta_information = additional_attributes['meta_information']
        _id = uuid.UUID(additional_attributes['id'])
        labels_path = join(path, 'labels.json')
        if exists(labels_path):
            with open(labels_path) as f:
                labels = json.load(f)
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
                       path_to_data: str, 
                       path_to_labels: Optional[str] = None,
                       keep_axis_orientation: bool = False, 
                       is_multichannel: bool = False,
                       name: str = ''):
        data, tif_metadata = read_image(path_to_data)
        if not keep_axis_orientation and len(data.shape) > 2:
            data = np.moveaxis(data, 0, 2)
        if path_to_data.endswith('ome.tif') or path_to_data.endswith('ome.tiff'):
            is_ome = True
        else:
            is_ome = False
        meta_information = {
            'path': path_to_data,
        }
        if path_to_labels is not None:
            meta_information['labels_path'] = path_to_labels
            with open(path_to_labels) as f:
                labels = [x.strip() for x in f.readlines()]
            if is_multichannel:
                ids = np.unique(data).astype(int)
                ids = sorted([x for x in ids if x != 0])
                labels = {x: y for (x,y) in zip(ids, labels)}
        else:
            labels = None
        return cls(data=data,
                   name=name, 
                   meta_information=meta_information,
                   is_ome=is_ome,
                   labels=labels,
                   keep_axis_orientation=keep_axis_orientation,
                   tif_metadata=tif_metadata,
                   is_multichannel=is_multichannel)