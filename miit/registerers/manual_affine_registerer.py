from dataclasses import dataclass
from typing import Any

import cv2
import numpy, numpy as np
import SimpleITK as sitk

from .base_registerer import Registerer, RegistrationResult

@dataclass
class ManualAffineRegistrationResult(RegistrationResult):
    
    transformation: numpy.ndarray
    target_size: tuple[int, int]

class ManualAffineRegisterer(Registerer):
    """
    Registration class that transforms spatial data using a transformation matrix.
    """

    name = 'ManualAffineRegisterer'
    
    def register_images(self, 
                        moving_img: numpy.ndarray, 
                        fixed_img: numpy.ndarray, 
                        **kwargs: dict) -> RegistrationResult:
        pass

    def transform_image(self, 
                        image: numpy.ndarray, 
                        transformation: 'ManualAffineRegistrationResult | numpy.ndarray', 
                        interpolation_mode: int | str, 
                        **kwargs: dict) -> numpy.ndarray:
        if isinstance(interpolation_mode, str):
            if interpolation_mode == 'NN':
                int_mode = sitk.sitkNearestNeighbor
            elif interpolation_mode == 'LINEAR':
                int_mode = sitk.sitkLinear
        transform = sitk.AffineTransform(2)
        if isinstance(transformation, 'ManualAffineRegistrationResult'):
            t_mat = transformation.transformation
            target_shape = transformation.target_size
        else:
            t_mat = transformation
            target_shape = image.shape[:2]
        transform.SetMatrix((t_mat[0,0], t_mat[0,1], t_mat[1,0], t_mat[1,1]))
        transform.SetTranslation((t_mat[0,2], t_mat[1,2]))
        # size = image.shape[:2]
        ref_img = sitk.GetImageFromArray(np.zeros((target_shape[0], target_shape[1])), True)
        sitk_image = sitk.GetImageFromArray(image, True)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(int_mode)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        transformed_image_sitk = resampler.Execute(sitk_image)
        transformed_image = sitk.GetArrayFromImage(transformed_image_sitk)    
        return transformed_image

    def transform_pointset(self, 
                           pointset: numpy.ndarray, 
                           transformation: ManualAffineRegistrationResult | numpy.ndarray, 
                           **kwargs: dict) -> numpy.ndarray:
        "Applies transformation to a pointset. Note, that the inverse of the transformation will be computed."
        transform = sitk.AffineTransform(2)
        if isinstance(transformation, ManualAffineRegistrationResult):
            t_mat = transformation.transformation
        else:
            t_mat = transformation        
        transform.SetMatrix((t_mat[0,0], t_mat[0,1], t_mat[1,0], t_mat[1,1]))
        transform.SetTranslation((t_mat[0,2], t_mat[1,2]))
        transform = transform.GetInverse()
        offset = 0.5
        warped_points = []
        for i in range(pointset.shape[0]):
            point = (pointset[i,0] - offset, pointset[i,1] - offset)
            warped_point = transform.TransformPoint(point)
            warped_points.append(warped_point)
        warped_pointset = np.array(warped_points)
        return warped_pointset


def get_center(img: numpy.ndarray) -> tuple[int, int]:
    w, h = img.shape[:2]
    w_c, h_c = int(w//2), int(h//2)
    return (w_c, h_c)


def get_rotation_matrix_around_center(img: numpy.ndarray, angle: float) -> numpy.ndarray:
    w_c, h_c = get_center(img)
    rot_mat = cv2.getRotationMatrix2D((h_c, w_c), angle, 1)
    return rot_mat    