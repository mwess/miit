from dataclasses import dataclass, field
from typing import Any, Dict, Union

from miit.spatial_data.spatial_omics.imzml import Imzml
from miit.spatial_data.spatial_omics.visium import Visium
from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics
from miit.spatial_data.image import (
    BaseImage,
    DefaultImage, 
    Pointset, 
    Annotation, 
    GeoJSONData
)


class SpatialDataLoaderException(Exception):
    pass


@dataclass
class SpatialDataLoader:
    
    class_map: Dict = field(init=False)
    
    def __post_init__(self):
        self.class_map = {}
        self.class_map[Imzml.get_type()] = Imzml
        self.class_map[Visium.get_type()] = Visium
        self.class_map[DefaultImage.get_type()] = DefaultImage
        self.class_map[Annotation.get_type()] = Annotation
        self.class_map[Pointset.get_type()] = Pointset
        self.class_map[GeoJSONData.get_type()] = GeoJSONData

    def load(self, 
             data_type: Any, 
             path: str,
             **kwargs: Dict):
        if data_type not in self.class_map:
            raise SpatialDataLoaderException(f'data_type {data_type} not found in loader.')
        return self.class_map[data_type].load(path, **kwargs)

    def add_class(self, clazz: Union[BaseImage, BaseSpatialOmics, Pointset, GeoJSONData]):
        if clazz.get_type() in self.class_map:
            raise SpatialDataLoaderException(f'class: {clazz.get_type()} already present in SpatialDataLoader.')
        self.class_map[clazz.get_type()] = clazz
        
    @staticmethod
    def load_default_loader():
        return SpatialDataLoader()
        
    
def load_spatial_omics_data(config):

    if config['data_type'] == 'ScilsExportImzml':
        return Imzml.from_config(config)
    elif config['data_type'] == 'Visium10X':
        return Visium.from_config(config)
    else:
        print(f"""data_type: {config['data_type']} not found.""")
    return None
