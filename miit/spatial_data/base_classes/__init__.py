from dataclasses import dataclass, field
from typing import Any, TypeVar

from miit.spatial_data.base_classes.base_imaging import (
    BaseImage,
    BasePointset,
    BaseSpatialOmics
)


# TODO: Remove
# # This might not be necessary, but lets keep it in for now.
# # def singleton(cls):
# #     """Makes a decorator for singletons.
    
# #     Taken from here: https://realpython.com/primer-on-python-decorators/#creating-singletons
# #     """
# #     @functools.wraps(cls)
# #     def wrapper_singleton(*args, **kwargs):
# #         if wrapper_singleton.instance is None:
# #             wrapper_singleton.instance = cls(*args, **kwargs)
# #         return wrapper_singleton.instance
# #     wrapper_singleton.instance = None
# #     return wrapper_singleton 


class ImagingDataLoaderException(Exception):
    pass


@dataclass
class ImagingDataLoader:
    """
    Class that handles loading of spatial omics data from stored data.

    Attributes:
        class_map (dict): Maps type to SO data type.
    """

    class_map: dict = field(init=False)

    def __post_init__(self):
        self.class_map = {}

    # TODO: data_type can probably be a str.
    def load(self,
             data_type: Any,
             path: str,
             **kwargs: dict):
        """
        Loads datatype from path.

        Args:
            data_type (Any): Type to load.
            path (str): Path to data.
        """
        if data_type not in self.class_map:
            raise ImagingDataLoaderException(f'data_type {data_type} not found in loader.')
        return self.class_map[data_type].load(path, **kwargs)

    def add_class(self, 
                  clazz,
                  force_overwrite: bool = False):
        """
        Adds class to data loader.
        """
        if force_overwrite or clazz.get_type() not in self.class_map:
            self.class_map[clazz.get_type()] = clazz
        # else:
        #     raise ImagingDataLoaderException(f'class: {clazz.get_type()} already present in SpatialOmicsDataLoader.')

    def get_registered_classes(self) -> list[str]:
        return list(self.class_map.keys())

    @staticmethod
    def load_default_loader():
        return ImagingDataLoader()


IMAGING_DATA_LOADER = ImagingDataLoader.load_default_loader()


T = TypeVar('T', covariant=True)


# @wrapt.decorator
def MIITobject(cls: T) -> T:
    if not (issubclass(cls, BaseImage) or issubclass(cls, BasePointset) or issubclass(cls, BaseSpatialOmics)):
        raise Exception(f'Object type {type(cls)} cannot be registered.')
    IMAGING_DATA_LOADER.add_class(cls)
    return cls