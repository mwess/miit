from dataclasses import dataclass
from os.path import exists, join
import uuid
from pathlib import Path
import shutil
from typing import Any, Dict, Optional

import itk
import numpy
import numpy as np

from .base_registerer import Registerer
    

# TODO: Either remove or rewrite, but Pointsetregisterer was not really functioning.

def get_default_translation_parametermap():
    parameter_object = itk.ParameterObject.New()
    default_map = parameter_object.GetDefaultParameterMap('rigid')
    default_map['Registration'] = [
        'MultiMetricMultiResolutionRegistration'
    ]
    original_metric = default_map['Metric']
    default_map['Metric'] = [original_metric[0], 'CorrespondingPointsEuclideanDistanceMetric']
    parameter_object.AddParameterMap(default_map)
    return parameter_object

def get_default_bspline_parametermap():
    parameter_object = itk.ParameterObject.New()
    

# TODO: Test and fix the registerers in this file.
def parse_transformix_pointset(result_point_set):
    parsed_results_points = []
    for i in range(result_point_set.shape[0]):
        output_point_str = ' '.join(list(result_point_set[i])).split(';')
        output_point_str = [x for x in output_point_str if 'OutputPoint' in x][0]
        idx1 = output_point_str.find('[') + 1
        idx2 = output_point_str.find(']')
        output_point_str = output_point_str[idx1:idx2].strip().split()
        x = float(output_point_str[0])
        y = float(output_point_str[1])
        parsed_results_points.append(np.array([x, y]))
    return np.array(parsed_results_points)

class SimpleElastixTransformation:
    
    parameter_object: Any
    reference_image: numpy.array

@dataclass
class SimpleElastixPointSetRegisterer(Registerer):
    
    tmp_dir: Optional[str] = None

    def __post_init__(self):
        if self.tmp_dir is None:
            self.__init_tmp_dir()
    
    def register_images(self,
                 moving_image: numpy.array,
                 fixed_image: numpy.array,
                 **kwargs: Dict):
        moving_pointset = kwargs['moving_pointset']
        fixed_pointset = kwargs['fixed_pointset']
        parameter_object = kwargs.get('parameter_object', get_default_translation_parametermap())
        log_to_console = kwargs.get('log_to_console', False)
        self.__init_tmp_dir()
        moving_pointset_path = join(self.tmp_dir, 'moving.txt')
        self.__write_pointset_to_file(moving_pointset_path, moving_pointset)
        fixed_pointset_path = join(self.tmp_dir, 'fixed.txt')
        self.__write_pointset_to_file(fixed_pointset_path, fixed_pointset)
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed_image, moving_image,
            fixed_point_set_file_name=fixed_pointset_path,
            moving_point_set_file_name=moving_pointset_path,
            log_to_console=log_to_console,
            parameter_object=parameter_object)
        # return
        # self.__remove_tmp_dir()
        transformation = SimpleElastixTransformation(result_transform_parameters, moving_image)
        return transformation, result_image
    
    def transform_image(self, 
                        image: numpy.array, 
                        transformation: SimpleElastixTransformation, 
                        interpolation: str,
                        **kwargs: Dict) -> numpy.array:
        if interpolation=='NN':
            transformation.SetParameter('FinalBSplineInterpolationOrder','0')
        else:
            transformation.SetParameter('FinalBSplineInterpolationOrder','1')
        transformed_image = itk.transformix_filter(image,
                                                   transformation)
        return transformed_image
    
    def transform_pointset(self, 
                           pointset: numpy.array, 
                           transformation: SimpleElastixTransformation,
                           **kwargs: Dict) -> numpy.array:
        # Coordinates should be in table form.
        # !!!!! IMPORTANT !!!! Its possible that coordinates need to be in 3D, but we try it out with 2D first.
        self.__init_tmp_dir()
        # Write coordinates to file
        fpath = join(self.tmp_dir, 'pointset_source.txt')
        self.__write_pointset_to_file(fpath, pointset)
        result_point_set = itk.transformix_pointset(
            transformation.reference_image, transformation.parameter_object,
            fixed_point_set_file_name=fpath,
            output_directory = self.tmp_dir)
        warped_points = parse_transformix_pointset(result_point_set)
        self.__remove_tmp_dir()
        return warped_points
    
    def __write_pointset_to_file(self, 
                                 path: str, 
                                 pointset):
        n_points = pointset.shape[0]
        with open(path, 'w') as f:
            f.write('points\n')
            f.write(f'{n_points}')
            for idx, row in pointset.iterrows():
                x = row['x']
                y = row['y']
                f.write(f'\n{x}\t{y}')
    
    def __init_tmp_dir(self):
        tmp_dir_path_base = 'tmp'
        tmp_dir_path = 'tmp'
        found_empty_dir = False
        failsafe_counter = 100000
        while not found_empty_dir and failsafe_counter > 0:
            failsafe_counter -= 1
            if not exists(tmp_dir_path):
                found_empty_dir = True
            else:
                uuid_ = str(uuid.uuid1())
                tmp_dir_path = f'{tmp_dir_path_base}_{uuid_}'
        Path(tmp_dir_path).mkdir(parents=True, exist_ok=True)
        self.tmp_dir = tmp_dir_path
        
    def __remove_tmp_dir(self):
        shutil.rmtree(self.tmp_dir)
        
    def load_from_config(cls, config: Dict):
        tmp_dir = config.get('tmp_dir', None)
        return cls(tmp_dir=tmp_dir)

        
class SimpleElastixSimpleRegisterer(Registerer):
    
    parameter_object: Any
    tmp_dir: Any
    
    def register(self, moving_image, fixed_image, **kwargs):
        parameter_object = kwargs.get('parameter_object', get_default_translation_parametermap())
        _, params = itk.elastix_registration_method(fixed_image, 
                                                    moving_image,
                                                    parameter_object,
                                                    log_to_console=False)
        transformation = SimpleElastixTransformation(params, moving_image)
        return transformation
    
    def transform(self, image, transformation, interpolation, args):
        if interpolation=='NN':
            transformation.parameter_object.SetParameter('FinalBSplineInterpolationOrder','0')
        else:
            transformation.parameter_object.SetParameter('FinalBSplineInterpolationOrder','1')
        transformed_image = itk.transformix_filter(image,
                                                   transformation.parameter_object)
        return transformed_image
    
    def transform_coordinates(self, pointset, transformation, **kwargs):
        # Coordinates should be in table form.
        # !!!!! IMPORTANT !!!! Its possible that coordinates need to be in 3D, but we try it out with 2D first.
        self.__init_tmp_dir()
        # Write coordinates to file
        fpath = join(self.tmp_dir, 'pointset_source.txt')
        self.__write_pointset_to_file(fpath, pointset)
        result_point_set = itk.transformix_pointset(
            transformation.reference_image, transformation.parameter_object,
            fixed_point_set_file_name=fpath,
            output_directory = self.tmp_dir)
        warped_points = parse_transformix_pointset(result_point_set)
        self.__remove_tmp_dir()
        return warped_points
        
    
    def __init_tmp_dir(self):
        tmp_dir_path_base = 'tmp'
        tmp_dir_path = 'tmp'
        found_empty_dir = False
        failsafe_counter = 100000
        while not found_empty_dir and failsafe_counter > 0:
            failsafe_counter -= 1
            if not exists(tmp_dir_path):
                found_empty_dir = True
            else:
                uuid_ = str(uuid.uuid1())
                tmp_dir_path = f'{tmp_dir_path_base}_{uuid_}'
        Path(tmp_dir_path).mkdir(parents=True, exists_ok=True)
        self.tmp_dir = tmp_dir_path
        
    
    def __write_pointset_to_file(self, path, pointset):
        n_points = pointset.shape[0]
        with open(path, 'w') as f:
            f.write('points\n')
            f.write(f'{n_points}')
            for idx, row in pointset.iterrows():
                x = row['x']
                y = row['y']
                f.write(f'\n{x}\t{y}')
        
    def __remove_tmp_dir(self):
        shutil.rmtree(self.tmp_dir)