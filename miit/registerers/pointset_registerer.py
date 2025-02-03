from dataclasses import dataclass
from typing import Any

import numpy, numpy as np
import skimage, skimage as ski

from miit.registerers.base_registerer import Registerer, RegistrationResult


def affine_transform(points: numpy.ndarray, 
                     transform: numpy.ndarray) -> numpy.ndarray:
    trans_points = (transform @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    return trans_points[:, 0:2]


@dataclass
class PointsetRegistrationResult(RegistrationResult):

    tform: skimage.transform._geometric.AffineTransform
    output_shape: tuple[int, int]


@dataclass
class PointsetRegisterer(Registerer):
    """Performs affine registration based on matching landmarks.
    """
    
    name = 'PointsetRegisterer'

    def register_images(self,
                        moving_image: numpy.ndarray,
                        fixed_image: numpy.ndarray,
                        moving_pointset: numpy.ndarray,
                        fixed_pointset: numpy.ndarray,
                        **kwargs: dict):
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
                        image: numpy.ndarray, 
                        tform: skimage.transform._geometric.AffineTransform,
                        output_shape: tuple[int, int],
                        interpolation_mode: str, 
                        **kwargs: dict) -> numpy.ndarray:
        order = 0 if interpolation_mode == 'NN' else 1
        return ski.transform.warp(image, 
                                  inverse_map=tform.inverse, 
                                  order=order,
                                  output_shape=output_shape)
    
    def transform_image(self, 
                        image: numpy.ndarray, 
                        transformation: RegistrationResult, 
                        interpolation_mode: str, 
                        **kwargs: dict) -> numpy.ndarray:
        order = 0 if interpolation_mode == 'NN' else 1
        warped_image = self._transform_image_affine(
            image,
            transformation.tform,
            transformation.output_shape,
            interpolation_mode
        )
        return warped_image

    def transform_pointset(self, 
                           pointset: numpy.ndarray, 
                           transformation: RegistrationResult, 
                           **kwargs: dict) -> numpy.ndarray:
        warped_pointset = []
        for i in range(pointset.shape[0]):
            ps_ = np.expand_dims(pointset[i,], 0)
            if np.isinf(ps_[0,0]) or np.isinf(ps_[0,1]):
                warped_pointset.append(ps_.squeeze())
                continue
            warped_ps_ = affine_transform(ps_, transformation.tform.params).squeeze()
            warped_pointset.append(warped_ps_)
        warped_pointset = np.array(warped_pointset)
        # warped_pointset = affine_transform(pointset, transformation.tform.params) 
        return warped_pointset
            
    @classmethod
    def load_from_config(cls, 
                         config: dict[str, Any] | None = None) -> Registerer:
        return cls()
