import abc
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy

from miit.registerers.base_registerer import Registerer, RegistrationResult


@dataclass(kw_only=True)
class BaseImage(abc.ABC):

    data: numpy.ndarray
    interpolation_mode: ClassVar[str]
    name: str = ''
    _id:uuid.UUID = field(init=False)
    meta_information: dict = field(default_factory=lambda: defaultdict(dict))

    def __post_init__(self) -> None:
        self._id = uuid.uuid1()

    def transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: dict) -> Any:
        image_transformed = registerer.transform_image(self.data, transformation, self.interpolation_mode, **kwargs)
        return image_transformed

    @abc.abstractmethod
    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        pass

    @abc.abstractmethod
    def resize(self, width: int, height: int):
        pass

    @abc.abstractmethod
    def resize(self, scaling_factor: float):
        pass

    @abc.abstractmethod
    def pad(self, padding: tuple[int, int, int, int], constant_values: int = 0):
        pass

    @abc.abstractmethod
    def flip(self, axis: int = 0):
        pass

    @abc.abstractmethod
    def apply_transform(self, registerer: Registerer, transformation: Any, **kwargs: dict) -> Any:
        pass

    @abc.abstractmethod
    def store(self, path: str):
        pass

    @abc.abstractmethod
    def get_resolution(self) -> float | None:
        pass

    @abc.abstractmethod
    def get_type(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def load(path: str) -> 'BaseImage':
        pass


@dataclass(kw_only=True)
class BasePointset(abc.ABC):

    data: Any
    name: str = ''
    _id:uuid.UUID = field(init=False)
    meta_information: dict = field(default_factory=lambda: defaultdict(dict))

    def __post_init__(self) -> None:
        self._id = uuid.uuid1()

    @abc.abstractmethod
    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        pass

    @abc.abstractmethod
    def resize(self, height: int, width: int):
        pass

    @abc.abstractmethod
    def rescale(self, scaling_factor: float):
        pass

    @abc.abstractmethod
    def pad(self, padding: tuple[int, int, int, int], constant_values: int = 0):
        pass

    @abc.abstractmethod
    def flip(self, axis: int = 0):
        pass

    @abc.abstractmethod
    def apply_transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: dict) -> Any:
        pass

    @abc.abstractmethod
    def store(self, path: str):
        pass

    @abc.abstractmethod
    def get_type(self) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def load(path: str) -> 'BasePointset':
        pass