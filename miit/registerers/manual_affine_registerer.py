from typing import Any

import cv2
import numpy, numpy as np
import SimpleITK as sitk

from .base_registerer import Registerer, RegistrationResult

class ManualAffineRegisterer(Registerer):

    name = 'ManualAffineRegisterer'
    
    def register_images(self, 
                        moving_img: numpy.ndarray, 
                        fixed_img: numpy.ndarray, 
                        **kwargs: dict) -> RegistrationResult:
        pass

    def transform_image(self, 
                        image: numpy.ndarray, 
                        transformation: numpy.ndarray, 
                        interpolation_mode: int, 
                        **kwargs: dict) -> numpy.ndarray:
        transform = sitk.AffineTransform(2)
        transform.SetMatrix((transformation[0,0], transformation[0,1], transformation[1,0], transformation[1,1]))
        transform.SetTranslation((transformation[0,2], transformation[1,2]))
        size = image.shape[:2]
        ref_img = sitk.GetImageFromArray(np.zeros((size[0], size[1])), True)
        sitk_image = sitk.GetImageFromArray(image, True)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(interpolation_mode)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        transformed_image_sitk = resampler.Execute(sitk_image)
        transformed_image = sitk.GetArrayFromImage(transformed_image_sitk)    
        return transformed_image

    def transform_pointset(self, 
                           pointset: numpy.ndarray, 
                           transformation: RegistrationResult, 
                           **kwargs: dict) -> numpy.ndarray:
        transform = sitk.AffineTransform(2)
        transform.SetMatrix((transformation[0,0], transformation[0,1], transformation[1,0], transformation[1,1]))
        transform.SetTranslation((transformation[0,2], transformation[1,2]))
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