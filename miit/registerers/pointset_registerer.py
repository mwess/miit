from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional

import numpy
import numpy as np
import skimage
import skimage as ski
import itk

from miit.registerers.base_registerer import Registerer, RegistrationResult


def affine_transform(points: numpy.array, 
                     transform: numpy.array) -> numpy.array:
    trans_points = (transform @ np.hstack((points, np.ones((points.shape[0], 1)))).T).T
    return trans_points[:, 0:2]


def load_default_bspline_parameters():
    return load_elastix_parameters_from_file('miit/registerers/bspline_parameters.txt')


def load_elastix_parameters_from_file(fpath):
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(fpath)
    return parameter_object


def parse_transformix_pointset(result_pointset) -> numpy.array:
    parsed_results_points = []
    for i in range(result_pointset.shape[0]):
        output_point_str = ' '.join(list(result_pointset[i])).split(';')
        output_point_str = [x for x in output_point_str if 'OutputPoint' in x][0]
        idx1 = output_point_str.find('[') + 1
        idx2 = output_point_str.find(']')
        output_point_str = output_point_str[idx1:idx2].strip().split()
        x = float(output_point_str[0])
        y = float(output_point_str[1])
        parsed_results_points.append(np.array([x, y]))
    return np.array(parsed_results_points)
    

def write_pointset_to_file(path: str,
                           pointset: numpy.array):
    n_points = pointset.shape[0]
    with open(path, 'w') as f:
        f.write('point\n')
        f.write(f'{n_points}')
        for idx in range(n_points):
            x = pointset[idx, 0]
            y = pointset[idx, 1]
            f.write(f'\n{x}\t{y}')


@dataclass
class PointsetRegistrationResult(RegistrationResult):

    tform: skimage.transform._geometric.AffineTransform
    output_shape: Tuple[int, int]


@dataclass
class PointsetRegisterer(Registerer):
    
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
