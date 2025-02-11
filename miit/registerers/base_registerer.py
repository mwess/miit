import abc
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy


@dataclass
class RegistrationResult:
    """Should be returned after for each pairwise registration."""
    pass


@dataclass
class Registerer(abc.ABC):
    """
    Provides methods that each registerer should implement. 
    """

    name: ClassVar[str]
    
    @abc.abstractmethod
    def register_images(self, 
                        moving_img: numpy.ndarray, 
                        fixed_img: numpy.ndarray, 
                        **kwargs: dict)-> 'RegistrationResult':
        pass

    @abc.abstractmethod
    def transform_pointset(self, 
                           pointset: numpy.ndarray, 
                           transformation: 'RegistrationResult', 
                           **kwargs: dict) -> numpy.ndarray:
        pass
    
    @abc.abstractmethod
    def transform_image(self, 
                        image: numpy.ndarray, 
                        transformation: 'RegistrationResult', 
                        interpolation_mode: int, **kwargs: dict) -> numpy.ndarray:
        pass
