from dataclasses import dataclass, field
from typing import Any, Dict, Union

from miit.spatial_data.spatial_omics.imzml import Imzml
from miit.spatial_data.spatial_omics.visium import Visium
from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics


class SpatialDataLoaderException(Exception):
    pass


@dataclass
class SpatialOmicsDataLoader:
    
    class_map: Dict = field(init=False)
    
    def __post_init__(self):
        self.class_map = {}
        self.class_map[Imzml.get_type()] = Imzml
        self.class_map[Visium.get_type()] = Visium

    def load(self, 
             data_type: Any, 
             path: str,
             **kwargs: Dict):
        if data_type not in self.class_map:
            raise SpatialDataLoaderException(f'data_type {data_type} not found in loader.')
        return self.class_map[data_type].load(path, **kwargs)

    def add_class(self, clazz: BaseSpatialOmics):
        if clazz.get_type() in self.class_map:
            raise SpatialDataLoaderException(f'class: {clazz.get_type()} already present in SpatialOmicsDataLoader.')
        self.class_map[clazz.get_type()] = clazz
        
    @staticmethod
    def load_default_loader():
        return SpatialOmicsDataLoader()