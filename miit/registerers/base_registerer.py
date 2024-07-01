import abc
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

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
                        **kwargs: Dict)-> 'RegistrationResult':
        pass

    @abc.abstractmethod
    def transform_pointset(self, 
                           pointset: numpy.array, 
                           transformation: 'RegistrationResult', 
                           **kwargs: Dict) -> numpy.array:
        pass
    
    @abc.abstractmethod
    def transform_image(self, 
                        image: numpy.array, 
                        transformation: 'RegistrationResult', 
                        interpolation_mode: str, **kwargs: Dict) -> numpy.array:
        pass

    @classmethod
    @abc.abstractmethod
    def load_from_config(cls, config: Dict[str, Any]) -> 'Registerer':
        pass

