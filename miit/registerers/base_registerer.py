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
                        moving_img: numpy.array, 
                        fixed_img: numpy.array, 
                        **kwargs: dict)-> 'RegistrationResult':
        pass

    @abc.abstractmethod
    def transform_pointset(self, 
                           pointset: numpy.array, 
                           transformation: 'RegistrationResult', 
                           **kwargs: dict) -> numpy.array:
        pass
    
    @abc.abstractmethod
    def transform_image(self, 
                        image: numpy.array, 
                        transformation: 'RegistrationResult', 
                        interpolation_mode: str, **kwargs: dict) -> numpy.array:
        pass
