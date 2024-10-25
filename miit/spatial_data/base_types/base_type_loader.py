from dataclasses import dataclass, field
from typing import Any

from miit.spatial_data.base_types import (
    Annotation,
    Image,
    BaseImage,
    BasePointset,
    Pointset,
    GeoJSONData,
    OMETIFFImage,
    OMETIFFAnnotation
)


class SpatialDataLoaderException(Exception):
    pass


@dataclass
class SpatialBaseDataLoader:
    
    class_map: dict = field(init=False)
    
    def __post_init__(self):
        self.class_map = {}
        self.class_map[Image.get_type()] = Image
        self.class_map[Annotation.get_type()] = Annotation
        self.class_map[Pointset.get_type()] = Pointset
        self.class_map[GeoJSONData.get_type()] = GeoJSONData
        self.class_map[OMETIFFImage.get_type()] = OMETIFFImage
        self.class_map[OMETIFFAnnotation.get_type()] = OMETIFFAnnotation

    def load(self, 
             data_type: str, 
             path: str,
             **kwargs: dict) -> Any:
        if data_type not in self.class_map:
            raise SpatialDataLoaderException(f'data_type {data_type} not found in loader.')
        return self.class_map[data_type].load(path, **kwargs)

    def add_class(self, clazz: BaseImage | BasePointset):
        if clazz.get_type() in self.class_map:
            raise SpatialDataLoaderException(f'class: {clazz.get_type()} already present in SpatialDataLoader.')
        self.class_map[clazz.get_type()] = clazz
        
    @staticmethod
    def load_default_loader():
        return SpatialBaseDataLoader()