from __future__ import annotations
import abc
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy

from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.distance_unit import DUnit


@dataclass(kw_only=True)
class BaseImage(abc.ABC):

    data: numpy.ndarray
    interpolation_mode: ClassVar[str]
    name: str = ''
    _id:uuid.UUID = field(init=False)
    meta_information: dict = field(default_factory=lambda: defaultdict(dict))
    resolution: tuple[DUnit, DUnit] = field(default_factory=lambda: [DUnit.default_dunit(), DUnit.default_dunit()])

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
    def rescale(self, scaling_factor: float | tuple[float, float]):
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
    def get_type(self) -> str:
        pass

    def scale_resolution(self, scale_factors: tuple[float, float]):
        res_w, res_h = self.resolution
        scale_w, scale_h = scale_factors
        res_w.scale(scale_w)
        res_h.scale(scale_h)
        self.resolution = (res_w, res_h)

    def scale_to_resolution(self, resolution: DUnit | tuple[DUnit, DUnit], align_units: bool = True):
        if isinstance(resolution, DUnit):
            resolution = (resolution, resolution)    
        res_w, res_h = self.resolution
        dst_w, dst_h = resolution
        conv_rate_w = 1 / res_w.get_conversion_factor(dst_w)
        conv_rate_h = 1 / res_h.get_conversion_factor(dst_h)
        self.rescale((conv_rate_w, conv_rate_h))
        if align_units:
            rw, rh = self.resolution
            self.resolution = (rw.convert_to_unit(dst_w.symbol), rh.convert_to_unit(dst_h.symbol))

    def align_resolution(self, target: BaseImage | BasePointset, align_units: bool = True):
        self.scale_to_resolution(target.resolution)

    def set_resolution(self, resolution: DUnit | tuple[DUnit]):
        if isinstance(resolution, DUnit):
            resolution = [resolution, resolution]
        self.resolution = resolution            

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
    resolution: tuple[DUnit, DUnit] = field(default_factory=lambda: [DUnit.default_dunit(), DUnit.default_dunit()])

    def __post_init__(self) -> None:
        self._id = uuid.uuid1()

    @abc.abstractmethod
    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        pass

    @abc.abstractmethod
    def resize(self, height: int, width: int):
        pass

    @abc.abstractmethod
    def rescale(self, scaling_factor: float | tuple[float, float]):
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

    def scale_resolution(self, scale_factors: tuple[float, float]):
        res_w, res_h = self.resolution
        scale_w, scale_h = scale_factors
        res_w.scale(scale_w)
        res_h.scale(scale_h)
        self.resolution = (res_w, res_h)    

    def scale_to_resolution(self, resolution: DUnit | tuple[DUnit, DUnit], align_units: bool = True):
        if isinstance(resolution, DUnit):
            resolution = (resolution, resolution)    
        res_w, res_h = self.resolution
        dst_w, dst_h = resolution
        conv_rate_w = 1 / res_w.get_conversion_factor(dst_w)
        conv_rate_h = 1 / res_h.get_conversion_factor(dst_h)
        conv_rate_w_f = float(conv_rate_w)
        conv_rate_h_f = float(conv_rate_h)
        self.rescale((conv_rate_w_f, conv_rate_h_f))
        if align_units:
            rw, rh = self.resolution
            self.resolution = (rw.convert_to_unit(dst_w.symbol), rh.convert_to_unit(dst_h.symbol))

    def align_resolution(self, target: BaseImage | BasePointset, align_units: bool = True):
        res_w, res_h = self.resolution
        dst_w, dst_h = target.resolution
        conv_rate_w = 1 / res_w.get_conversion_factor(dst_w)
        conv_rate_h = 1 / res_h.get_conversion_factor(dst_h)
        conv_rate_w = float(conv_rate_w)
        conv_rate_h = float(conv_rate_h)
        self.rescale((conv_rate_w, conv_rate_h))
        self.resolution = target.resolution
        
    def set_resolution(self, resolution: DUnit | tuple[DUnit]):
        if isinstance(resolution, DUnit):
            resolution = [resolution, resolution]
        self.resolution = resolution              