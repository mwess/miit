from dataclasses import dataclass
from os.path import join, exists
from pathlib import Path
import shutil
from typing import Any, Dict, Tuple, Optional, Union
import uuid

import numpy
import numpy as np
import skimage
import skimage as ski
from skimage.filters.rank import majority
from skimage.morphology import disk
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
    do_nonrigid: bool
    nr_pre_reg_downscaling_factor: float
    nr_gaussian_sigma: float
    nr_parameter_object: Any# itk.elxParameterObjectPython.elastixParameterObject
    output_shape: Tuple[int, int]


@dataclass
class PointsetRegisterer(Registerer):
    
    name = 'PointsetRegisterer'

    def register_images(self,
                        moving_image: numpy.array,
                        fixed_image: numpy.array,
                        moving_pointset: numpy.array,
                        fixed_pointset: numpy.array,
                        enable_nonrigid: bool = True,
                        nr_pre_reg_downscaling_factor: float = 1/4,
                        nr_gaussian_sigma: float = 1,
                        # bspline_parameter_map: Optional[Union[str, itk.elxParameterObjectPython.elastixParameterObject]] = None,
                        bspline_parameter_map = None,
                        **kwargs: Dict):
        

        # Assumes that pointsets are already numpy array
        # Performs affine registration based on matching pointsets
        tform = ski.transform.AffineTransform()
        tform.estimate(moving_pointset, fixed_pointset)
        output_shape = (fixed_image.shape[0], fixed_image.shape[1])

        if enable_nonrigid:
            if bspline_parameter_map is None:
                bspline_parameter_map = load_default_bspline_parameters()
            elif isinstance(bspline_parameter_map, str):
                bspline_parameter_map = load_elastix_parameters_from_file(bspline_parameter_map)
            else:
                pass
                # Assume that otherwise the correct parameterobject has been supplied. 
                # TODO: Add check for isinstance later.!
            warped_image  = self._transform_image_affine(moving_image, tform, output_shape, 'LINEAR')
            warped_image_small = skimage.transform.rescale(warped_image, nr_pre_reg_downscaling_factor, channel_axis=len(warped_image.shape)-1)
            fixed_image_small = skimage.transform.rescale(fixed_image, nr_pre_reg_downscaling_factor, channel_axis=len(fixed_image.shape)-1)
            warped_image_small = skimage.color.rgb2gray(warped_image_small)
            fixed_image_small = skimage.color.rgb2gray(fixed_image_small)
            warped_image_small = skimage.filters.gaussian(warped_image_small, sigma=nr_gaussian_sigma)
            fixed_image_small = skimage.filters.gaussian(fixed_image_small, sigma=nr_gaussian_sigma)
            warped_image_small = itk.image_from_array(warped_image_small)
            fixed_image_small = itk.image_from_array(fixed_image_small)
            # TODO: Should this be done with pointsets as well
            result_image_bspline, nr_result_transform_parameters = itk.elastix_registration_method(
                fixed_image_small, warped_image_small,
                parameter_object=bspline_parameter_map,
                log_to_console=False)
        else:
            nr_result_transform_parameters = None
        
        return PointsetRegistrationResult(
            tform=tform,
            do_nonrigid=enable_nonrigid,
            nr_pre_reg_downscaling_factor=nr_pre_reg_downscaling_factor,
            nr_gaussian_sigma=nr_gaussian_sigma,
            nr_parameter_object=nr_result_transform_parameters,
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
        if transformation.do_nonrigid:
            transformation.nr_parameter_object.SetParameter('FinalBSplintInterpolationOrder', str(order))
            channel_axis = 2 if len(warped_image.shape) == 3 else None
            warped_image = ski.transform.rescale(warped_image, 
                                                 transformation.nr_pre_reg_downscaling_factor, 
                                                 channel_axis=channel_axis, 
                                                 order=order)
            warped_channels = []
            if len(warped_image.shape) == 3:
                for channel_idx in range(len(warped_image.shape)):
                    img = itk.image_from_array(warped_image[:,:,channel_idx])
                    warped_image_channel = itk.transformix_filter(img, transformation.nr_parameter_object)
                    warped_channels.append(itk.GetArrayFromImage(warped_image_channel))
                warped_image = np.dstack(warped_channels)
            else:
                # Deal with integer in 2d images.
                if np.issubdtype(warped_image.dtype, np.integer):
                    img_dtype = warped_image.dtype
                else:
                    img_dtype = None
                # warped_image_itk = itk.transformix_filter(itk.image_from_array(warped_image.astype(np.float32)), transformation.nr_parameter_object)
                warped_image_itk = itk.transformix_filter(itk.image_from_array(warped_image), transformation.nr_parameter_object)
                warped_image = itk.GetArrayFromImage(warped_image_itk)

                # return warped_image
                if img_dtype is not None:
                    # Do rank majority filtering to filter out holes after warping masks.
                    warped_image = majority(warped_image, disk(5))
                    # warped_image = np.round(warped_image).astype(img_dtype)
            warped_image = ski.transform.resize(warped_image, transformation.output_shape, order=order)
        return warped_image

    def transform_pointset(self, 
                           pointset: numpy.array, 
                           transformation: RegistrationResult, 
                           **kwargs: Dict) -> numpy.array:
        warped_pointset = affine_transform(pointset, transformation.tform.params) 
        if transformation.do_nonrigid:
            self.__init_tmp_dir()
            # Write coordinates to file
            fpath = join(self.tmp_dir, 'pointset_source.txt')
            warped_pointset = warped_pointset * transformation.nr_pre_reg_downscaling_factor
            write_pointset_to_file(fpath, warped_pointset)
            instance = itk.image_view_from_array(np.zeros(transformation.output_shape))
            transformation.nr_parameter_object.SetParameter('FinalBSplintInterpolationOrder', '1')
            result_point_set = itk.transformix_pointset(
                instance, transformation.nr_parameter_object,
                fixed_point_set_file_name=fpath,
                output_directory=self.tmp_dir,
                log_to_console=False)
            warped_pointset = parse_transformix_pointset(result_point_set)
            warped_pointset = warped_pointset / transformation.nr_pre_reg_downscaling_factor
            self.__remove_tmp_dir()
        return warped_pointset
            
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
    
    @classmethod
    def load_from_config(cls, 
                         config: Dict[str, Any]) -> Registerer:
        return cls()
