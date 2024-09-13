from typing import Any, Dict

import cv2
import numpy, numpy as np
import SimpleITK as sitk

from .base_registerer import Registerer, RegistrationResult

class ManualAffineRegisterer(Registerer):

    name = 'ManualAffineRegisterer'
    
    def register_images(self, moving_img: numpy.array, fixed_img: numpy.array, **kwargs: Dict) -> RegistrationResult:
        pass

    def transform_image(self, 
                        image: numpy.array, 
                        transformation: numpy.array, 
                        interpolation_mode: str, **kwargs: Dict) -> numpy.array:
        if interpolation_mode == 'LINEAR':
            sitk_interpolation = sitk.sitkLinear
        else:
            sitk_interpolation = sitk.sitkNearestNeighbor
        transform = sitk.AffineTransform(2)
        transform.SetMatrix((transformation[0,0], transformation[0,1], transformation[1,0], transformation[1,1]))
        transform.SetTranslation((transformation[0,2], transformation[1,2]))
        size = image.shape[:2]
        ref_img = sitk.GetImageFromArray(np.zeros((size[0], size[1])), True)
        sitk_image = sitk.GetImageFromArray(image, True)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(sitk_interpolation)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        transformed_image_sitk = resampler.Execute(sitk_image)
        transformed_image = sitk.GetArrayFromImage(transformed_image_sitk)    
        return transformed_image

    def transform_pointset(self, 
                           pointset: numpy.array, 
                           transformation: RegistrationResult, **kwargs: Dict) -> numpy.array:
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
    
    @classmethod
    def load_from_config(cls, config: Dict[str, Any]) -> Registerer:
        return cls()

def get_center(img: numpy.array):
    w, h = img.shape[:2]
    w_c, h_c = int(w//2), int(h//2)
    return (w_c, h_c)

def get_rotation_matrix_around_center(img: numpy.array, angle: float):
    w_c, h_c = get_center(img)
    rot_mat = cv2.getRotationMatrix2D((w_c, h_c), angle, 1)
    return rot_mat    