from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy, numpy as np
import skimage, skimage as ski
import itk

from miit.registerers.base_registerer import Registerer, RegistrationResult


def affine_transform(points: numpy.array, 
                     transform: numpy.array) -> numpy.array:
    trans_points = (transform @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    return trans_points[:, 0:2]


@dataclass
class PointsetRegistrationResult(RegistrationResult):

    tform: skimage.transform._geometric.AffineTransform
    output_shape: Tuple[int, int]


@dataclass
class PointsetRegisterer(Registerer):
    """Performs affine registration based on matching landmarks.
    """
    
    name = 'PointsetRegisterer'

    def register_images(self,
                        moving_image: numpy.array,
                        fixed_image: numpy.array,
                        moving_pointset: numpy.array,
                        fixed_pointset: numpy.array,
                        **kwargs: Dict):
        # Assumes that pointsets are already numpy array
        # Performs affine registration based on matching pointsets
        tform = ski.transform.AffineTransform()
        tform.estimate(moving_pointset, fixed_pointset)
        output_shape = (fixed_image.shape[0], fixed_image.shape[1])
        
        return PointsetRegistrationResult(
            tform=tform,
            output_shape=output_shape
        )
    
    def _transform_image_affine(self, 
                        image: numpy.array, 
                        tform: skimage.transform._geometric.AffineTransform,
                        output_shape: Tuple[int, int],
                        interpolation_mode: str, 
                        **kwargs: Dict) -> numpy.array:
        order = 0 if interpolation_mode == 'NN' else 1
        return ski.transform.warp(image, 
                                  inverse_map=tform.inverse, 
                                  order=order,
                                  output_shape=output_shape)
    
    def transform_image(self, 
                        image: numpy.array, 
                        transformation: RegistrationResult, 
                        interpolation_mode: str, 
                        **kwargs: Dict) -> numpy.array:
        order = 0 if interpolation_mode == 'NN' else 1
        warped_image = self._transform_image_affine(
            image,
            transformation.tform,
            transformation.output_shape,
            interpolation_mode
        )
        return warped_image

    def transform_pointset(self, 
                           pointset: numpy.array, 
                           transformation: RegistrationResult, 
                           **kwargs: Dict) -> numpy.array:
        warped_pointset = affine_transform(pointset, transformation.tform.params) 
        return warped_pointset
            
    @classmethod
    def load_from_config(cls, 
                         config: Optional[Dict[str, Any]] = None) -> Registerer:
        return cls()
