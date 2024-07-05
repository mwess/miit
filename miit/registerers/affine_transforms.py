import numpy
import numpy as np
import SimpleITK as sitk


def affine_image_transform(image: numpy.array, 
                           matrix: numpy.array,
                           sitk_interpolation: int) -> numpy.array:
    transform = sitk.AffineTransform(2)
    transform.SetMatrix((matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1]))
    transform.SetTranslation((matrix[0,2], matrix[1,2]))
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


def affine_pointset_transform(points: numpy.array,
                              matrix: numpy.array,
                              sitk_interpolation: int) -> numpy.array:
    pass