import abc
from dataclasses import dataclass, field
from typing import Any
import uuid

from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.spatial_data.base_types.annotation import Annotation


@dataclass
class BaseSpatialOmics(abc.ABC):
    
    _id: uuid.UUID = field(init=False)
    background: int

    @abc.abstractproperty
    def ref_mat(self):
        pass

    @ref_mat.setter
    def ref_mat(self, ref_mat: Annotation):
        pass
    
    @abc.abstractmethod
    def apply_transform(self, registerer: Registerer, transformation: RegistrationResult, args: dict[Any, Any] | None = None) -> 'BaseSpatialOmics':
        pass

    @abc.abstractmethod
    def pad(self, padding: tuple[int, int, int, int]):
        pass

    @abc.abstractmethod
    def resize(self, height: int, width: int):
        pass

    @abc.abstractmethod
    def rescale(self, scaling_factor: float):
        pass

    @abc.abstractmethod
    def crop(self, x1: int, x2: int, y1: int, y2: int):
        pass

    @abc.abstractmethod
    def flip(self, axis: int = 0):
        pass
    
    @abc.abstractmethod
    def store(self, directory: str):
        pass

    @abc.abstractmethod
    def get_spec_to_ref_map(self, reverse=False):
        pass

    @abc.abstractmethod
    def get_type(self):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, directory: str):
        pass