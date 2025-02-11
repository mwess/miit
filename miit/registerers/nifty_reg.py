from dataclasses import dataclass
import os
from os.path import join, exists
import shlex
import shutil
import subprocess
from typing import Any

import cv2
import niftyreg
import numpy, numpy as np
import SimpleITK, SimpleITK as sitk

from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.utils import derive_unique_directory


def _get_max_dim(img: numpy.ndarray) -> int:
    x, y = img.shape[:2]
    return x if x > y else y    


def _resize_image_simple_sitk(image: numpy.ndarray | sitk.Image, 
                             res: tuple[int, int], 
                             out_type: type = np.float32,
                             interpolation: int = cv2.INTER_NEAREST):
    if isinstance(image, sitk.Image):
        img_np = sitk.GetArrayFromImage(image)
    else:
        img_np = image
    new_img_np = cv2.resize(img_np.astype(np.float32), (res[1], res[0]), 0, 0, interpolation=interpolation)
    return sitk.GetImageFromArray(new_img_np.astype(np.float32))


@dataclass
class NiftyRegRegistrationResult(RegistrationResult):
    transform: SimpleITK.Transform
    downscaling_factor: float 
    target_size: tuple[int, int]
    orig_target_size: tuple[int, int]
    cmdln_returns: list[Any]

NIFTY_MAX_DIM_SIZE: int = 2048

@dataclass
class NiftyRegWrapper(Registerer):
    """Wrapper for NiftyReg registration algorithm (https://github.com/KCL-BMEIS/niftyreg).
    At the moment only rigid registration is supported.
    """

    name: str = 'NiftyRegWrapper'
    path_to_nifty_reg_aladin: str = ''
    path_to_nifty_reg_f3d: str = ''
    path_to_nifty_reg_resample: str = ''
    path_to_nifty_reg_transform: str = ''
    
    def register_images(self, 
                        moving_img: numpy.ndarray, 
                        fixed_img: numpy.ndarray, 
                        tmp_dir: str = 'tmp',
                        affine: bool = True,
                        nonrigid: bool = True,
                        affine_as_rigid: bool = True,
                        remove_tmp_dir: bool = True,
                        **kwargs: dict) -> RegistrationResult:
    # Export images
        if not affine and not nonrigid:
            raise Exception('Affine and nonrigid registration are disabled. Choose at least one to continue.')
        tmp_dir = derive_unique_directory(tmp_dir)
        moving_img_sitk = sitk.GetImageFromArray(moving_img)
        fixed_img_sitk = sitk.GetImageFromArray(fixed_img)
        max_dim_moving = _get_max_dim(moving_img)
        max_dim_fixed = _get_max_dim(fixed_img)
        src_fixed_img_shape = fixed_img.shape[:2]
        max_dim = max_dim_moving if max_dim_moving > max_dim_fixed else max_dim_fixed
        if max_dim <= NIFTY_MAX_DIM_SIZE:
            downscaling_factor = 1
        else:
            downscaling_factor = max_dim / NIFTY_MAX_DIM_SIZE
        if downscaling_factor > 1:
            fixed_half_size = (int(fixed_img.shape[0] / downscaling_factor), int(fixed_img.shape[1] / downscaling_factor))
            fixed_image_half_res = _resize_image_simple_sitk(fixed_img_sitk, fixed_half_size)
            moving_half_size = (int(moving_img.shape[0] / downscaling_factor), int(moving_img.shape[1] / downscaling_factor))
            moving_image_half_res = _resize_image_simple_sitk(moving_img_sitk, moving_half_size)
        else:
            fixed_image_half_res = fixed_img_sitk
            moving_image_half_res = moving_img_sitk
        fixed_img_shape = fixed_image_half_res.GetSize()[:2]
        # Write to file
        cmd_returns = []
        if not exists(tmp_dir):
            os.mkdir(tmp_dir)
        fixed_full_path = join(tmp_dir, 'fixed_full.nii.gz')
        moving_full_path = join(tmp_dir, 'moving_full.nii.gz')
        fixed_path = join(tmp_dir, 'fixed.nii.gz')
        moving_path = join(tmp_dir, 'moving.nii.gz')
        warped_image_path = join(tmp_dir, 'warped_image.nii.gz')
        warped_f3d_image_path = join(tmp_dir, 'warped_f3d_image.nii.gz')
        affine_transform_path = join(tmp_dir, 'Affine.txt')
        f3d_transform_path = join(tmp_dir, 'f3d_transform.nii.gz')
        f3d_transform_pp_path = join(tmp_dir, 'f3d_transform_pp.nii.gz')
        sitk.WriteImage(fixed_image_half_res, fixed_path)
        sitk.WriteImage(moving_image_half_res, moving_path)
        sitk.WriteImage(fixed_img_sitk, fixed_full_path)
        sitk.WriteImage(moving_img_sitk, moving_full_path)

        # Perform registrations
        if affine:
            cmd_aladin = [
                self.path_to_nifty_reg_aladin,
                '-ref', fixed_path,
                '-flo', moving_path,
                '-res', warped_image_path,
                '-aff', affine_transform_path,
                '-speeeeed'
            ]
            if affine_as_rigid:
                cmd_aladin.append('-rigOnly')
            cmd_aladin = ' '.join(cmd_aladin)
            ret_aladin = subprocess.run(shlex.split(cmd_aladin), capture_output=True)
            if len(ret_aladin.stderr) != 0:
                print('Error during affine registration:')
                print(ret_aladin.stderr.decode('utf-8'))
            cmd_returns.append(ret_aladin)    
        else:
            affine_transform_path = None    
        if nonrigid:
            cmd_f3d = [
                self.path_to_nifty_reg_f3d,
                '-ref', fixed_path,
                '-flo', moving_path,
                '-res', warped_f3d_image_path,
                '-cpp', f3d_transform_path
            ]
            if affine_transform_path is not None:
                cmd_f3d += ['-aff', affine_transform_path]
            cmd_f3d = ' '.join(cmd_f3d)
            ret_f3d = subprocess.run(shlex.split(cmd_f3d), capture_output=True)         
            cmd_returns.append(ret_f3d)
            # Postprocessing into displacement field
            cmd_post_processing = [
                self.path_to_nifty_reg_transform,
                '-ref', fixed_path,
                '-disp', f3d_transform_path, f3d_transform_pp_path
            ]       
            cmd_post_processing = ' '.join(cmd_post_processing)
            ret_pp = subprocess.run(shlex.split(cmd_post_processing), capture_output=True)
            cmd_returns.append(ret_pp)
        if nonrigid:
            disp_fields = sitk.Cast(sitk.ReadImage(f3d_transform_pp_path), sitk.sitkVectorFloat64)
            transform = sitk.DisplacementFieldTransform(2)
            transform.SetDisplacementField(disp_fields)            
        else:
            # Assume affine transform otherwise
            with open(affine_transform_path) as f:
                aff_params = f.readlines()
                aff_params = [x.strip() for x in aff_params]
            m = _string_mat_to_mat(aff_params)
            transform = sitk.AffineTransform(2)
            transform.SetMatrix((m[0, 0], m[0, 1], m[1, 0], m[1, 1]))
            transform.SetTranslation((m[0, 3], m[1, 3]))            
        if exists(tmp_dir) and remove_tmp_dir:
            shutil.rmtree(tmp_dir)
        registration_result = NiftyRegRegistrationResult(
            transform=transform, 
            downscaling_factor=downscaling_factor,
            target_size=fixed_img_shape,
            orig_target_size=src_fixed_img_shape,
            cmdln_returns=cmd_returns)
        return registration_result
    
    def transform_image(self, 
                        image: numpy.ndarray, 
                        transformation: RegistrationResult, 
                        interpolation_mode: int | str, 
                        tmp_directory: str = 'tmp', 
                        keep_src_dtype: bool = True,
                        **kwargs: dict) -> numpy.ndarray:
        src_dtype = image.dtype
        tmp_directory = derive_unique_directory(tmp_directory)
        if tmp_directory != '' and not os.path.exists(tmp_directory):
            os.mkdir(tmp_directory)
        sitk_image = sitk.GetImageFromArray(image)
        downscaling_factor = transformation.downscaling_factor
        if downscaling_factor > 1:
            moving_half_size = (int(image.shape[0] / downscaling_factor), int(image.shape[1] / downscaling_factor))
            sitk_image_half_res = _resize_image_simple_sitk(sitk_image, moving_half_size)
        else:
            sitk_image_half_res = sitk_image
        transform = transformation.transform
        
        if isinstance(interpolation_mode, str):
            if interpolation_mode == 'NN':
                int_mode = sitk.sitkNearestNeighbor
            elif interpolation_mode == 'LINEAR':
                int_mode = sitk.sitkLinear
        else:
            int_mode = interpolation_mode
        ref_img = sitk.GetImageFromArray(np.zeros((transformation.target_size[0], transformation.target_size[1])), True)
        sitk_image = sitk.GetImageFromArray(image, True)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref_img)
        resampler.SetInterpolator(int_mode)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(transform)
        warped_image_sitk = resampler.Execute(sitk_image_half_res)
        warped_image = sitk.GetArrayFromImage(warped_image_sitk)
        warped_image = _resize_image_simple_sitk(warped_image, transformation.orig_target_size, interpolation=int_mode)

        if os.path.exists(tmp_directory):
            shutil.rmtree(tmp_directory)
        warped_image = sitk.GetArrayFromImage(warped_image)
        if keep_src_dtype:
            warped_image = warped_image.astype(src_dtype)
        return warped_image
    
    def transform_pointset(self, 
                           pointset: numpy.ndarray, 
                           transformation: RegistrationResult, 
                           **kwargs: dict) -> numpy.ndarray:
        # Check for off-by-one errors.
        transform = transformation.transform
        downscaling_factor = transformation.downscaling_factor
        if downscaling_factor > 1:
            pointset = pointset/downscaling_factor
        pointset -= 0.5
        warped_points = []
        for i in range(pointset.shape[0]):
            point = (pointset[i,0], pointset[i,1])
            warped_point = transform.TransformPoint(point)
            warped_points.append(warped_point)
        warped_pointset = np.array(warped_points)
        if downscaling_factor > 1:
            warped_pointset = warped_pointset * downscaling_factor
        return warped_pointset

    @classmethod
    def init_registerer(cls, nifty_directory: str | None = None) -> Registerer:
        if nifty_directory is None:
            nifty_directory = str(niftyreg.bin_path)
        path_to_nifty_reg_aladin = join(nifty_directory, 'reg_aladin')
        path_to_nifty_reg_f3d = join(nifty_directory, 'reg_f3d')
        path_to_nifty_reg_resample = join(nifty_directory, 'reg_resample')
        path_to_nifty_reg_transform = join(nifty_directory, 'reg_transform')
        return cls(path_to_nifty_reg_aladin=path_to_nifty_reg_aladin,
                  path_to_nifty_reg_f3d=path_to_nifty_reg_f3d,
                  path_to_nifty_reg_resample=path_to_nifty_reg_resample,
                  path_to_nifty_reg_transform=path_to_nifty_reg_transform)            
    

def _string_mat_to_mat(str_mat: str) -> numpy.ndarray:
    mat = [[float(y) for y in x.split()] for x in str_mat]
    return np.array(mat)