from dataclasses import dataclass
import os
from os.path import join, exists
import shlex
import shutil
import subprocess
from typing import Any, Dict, List, Tuple, Optional

import cv2
import niftyreg
import numpy
import numpy as np
import SimpleITK as sitk

from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.utils import derive_unique_directory


def get_max_dim(img: numpy.array):
    x, y = img.shape[:2]
    return x if x > y else y    


def resize_image_simple_sitk(image, res, out_type=np.float32):
    img_np = sitk.GetArrayFromImage(image)
    new_img_np = cv2.resize(img_np.astype(np.float32), (res[1], res[0]), 0, 0, interpolation=cv2.INTER_NEAREST)
    return sitk.GetImageFromArray(new_img_np.astype(np.float32))


@dataclass
class NiftiRegRegistrationResult(RegistrationResult):
    rigid_transform: str
    downscaling_factor: float 
    target_size: Tuple[int, int]
    cmdln_returns: List[Any]


@dataclass
class NiftyRegWrapper(Registerer):
    """Wrapper for NiftyReg registration algorithm (https://github.com/KCL-BMEIS/niftyreg).
    At the moment only rigid registration is supported.
    """

    name: str = 'NiftyRegWrapper'
    path_to_nifty_reg_aladin: str = ''
    path_to_nifty_reg_f3d: str = ''
    path_to_nifty_reg_resample: str = ''
    
    def register_images(self, 
                        moving_img: numpy.array, 
                        fixed_img: numpy.array, 
                        tmp_dir: str = 'tmp',
                        reg_mode: str = 'rigid',
                        **kwargs: Dict) -> RegistrationResult:
    # Export images
        tmp_dir = derive_unique_directory(tmp_dir)
        nifty_max_dim_size = 2048
        moving_img_sitk = sitk.GetImageFromArray(moving_img)
        fixed_img_sitk = sitk.GetImageFromArray(fixed_img)
        max_dim_moving = get_max_dim(moving_img)
        max_dim_fixed = get_max_dim(fixed_img)
        max_dim = max_dim_moving if max_dim_moving > max_dim_fixed else max_dim_fixed
        if max_dim <= nifty_max_dim_size:
            downscaling_factor = 1
        else:
            downscaling_factor = max_dim / nifty_max_dim_size
        if downscaling_factor > 1:
            fixed_half_size = (int(fixed_img.shape[0] / downscaling_factor), int(fixed_img.shape[1] / downscaling_factor))
            fixed_image_half_res = resize_image_simple_sitk(fixed_img_sitk, fixed_half_size)
            moving_half_size = (int(moving_img.shape[0] / downscaling_factor), int(moving_img.shape[1] / downscaling_factor))
            moving_image_half_res = resize_image_simple_sitk(moving_img_sitk, moving_half_size)
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
        warped_rig_image_path = join(tmp_dir, 'warped_rig_image.nii.gz')
        rigid_transform_path = join(tmp_dir, 'Rigid.txt')
        sitk.WriteImage(fixed_image_half_res, fixed_path)
        sitk.WriteImage(moving_image_half_res, moving_path)
        sitk.WriteImage(fixed_img_sitk, fixed_full_path)
        sitk.WriteImage(moving_img_sitk, moving_full_path)
    
        # Perform registrations
        cmd_aladin_rig = f"""'{self.path_to_nifty_reg_aladin}' -ref '{fixed_path}' -flo '{moving_path}' -res '{warped_rig_image_path}' -aff '{rigid_transform_path}' -rigOnly"""
        ret_aladin_rig = subprocess.run(shlex.split(cmd_aladin_rig), capture_output=True)
        # cmd_aladin_rig = ['aladin', '-ref', f'{fixed_path}', '-flo', f'{moving_path}', '-res', f'{warped_rig_image_path}', '-aff', f'{rigid_transform_path}', '-rigOnly']
        # ret_aladin_rig = nifity_reg.main(args=cmd_aladin_rig)
        if len(ret_aladin_rig.stderr) != 0:
            print('Error during rigid registration:')
            print(ret_aladin_rig.stderr.decode('utf-8'))
        cmd_returns.append(ret_aladin_rig)    
        # Load resampled 
        warped_rig_image = sitk.ReadImage(warped_rig_image_path)
        target_img_size = sitk.GetArrayFromImage(moving_img_sitk).shape
        warped_rig_image = resize_image_simple_sitk(warped_rig_image, target_img_size)
        with open(rigid_transform_path) as f:
            rigid_transform = f.readlines()
            rigid_transform = [x.strip() for x in rigid_transform]
        if exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        registration_result = NiftiRegRegistrationResult(
            rigid_transform=rigid_transform, 
            downscaling_factor=downscaling_factor,
            target_size=fixed_img_shape,
            cmdln_returns=cmd_returns)
        return registration_result
    
    def transform_image(self, 
                        image: numpy.array, 
                        transformation: RegistrationResult, 
                        interpolation_mode: str, 
                        tmp_directory: str = 'tmp', 
                        **kwargs: Dict) -> numpy.array:
        tmp_directory = derive_unique_directory(tmp_directory)
        if tmp_directory != '' and not os.path.exists(tmp_directory):
            os.mkdir(tmp_directory)
        sitk_image = sitk.GetImageFromArray(image)
        downscaling_factor = transformation.downscaling_factor
        if downscaling_factor > 1:
            moving_half_size = (int(image.shape[0] / downscaling_factor), int(image.shape[1] / downscaling_factor))
            sitk_image_half_res = resize_image_simple_sitk(sitk_image, moving_half_size)
        else:
            sitk_image_half_res = sitk_image
        moving_path = join(tmp_directory, 'image.nii.gz')
        ref_path = join(tmp_directory, 'ref_image.nii.gz')
        output_path = join(tmp_directory, 'warped_image.nii.gz')
        empty_image = sitk.GetImageFromArray(np.zeros(transformation.target_size))
        sitk.WriteImage(empty_image, ref_path)
        sitk.WriteImage(sitk_image_half_res, moving_path)
        transform_path = join(tmp_directory, 'rigid_transform.txt')
        with open(transform_path, 'w') as f:
            f.write('\n'.join(transformation.rigid_transform))
        if interpolation_mode == 'LINEAR':
            interpolation = 1
        else:
            interpolation = 0
        cmd_resample = f"""'{self.path_to_nifty_reg_resample}' -ref '{ref_path}' -flo '{moving_path}' -res '{output_path}'  -trans '{transform_path}' -inter {interpolation}"""
        ret_resample = subprocess.run(shlex.split(cmd_resample), capture_output=True)
        if len(ret_resample.stderr) != 0:
            print('Error during warping:')
            print(ret_resample.stderr.decode('utf-8'))
        warped_image = sitk.ReadImage(output_path)
        if downscaling_factor > 1:
            target_img_size = image.shape
            warped_image = resize_image_simple_sitk(warped_image, target_img_size)
        if os.path.exists(tmp_directory):
            shutil.rmtree(tmp_directory)
        return sitk.GetArrayFromImage(warped_image)
    
    def transform_pointset(self, pointset: numpy.array, transformation: RegistrationResult, **kwargs: Dict) -> numpy.array:
        # TODO: Implement that.
        pass
    
    @classmethod
    def load_from_config(cls, config: Optional[Dict[str, Any]] = None) -> Registerer:
        # We dont use the interface of the niftyreg package directly to better control output to the console.
        nifti_directory = str(niftyreg.bin_path)
        path_to_nifty_reg_aladin = join(nifti_directory, 'reg_aladin')
        path_to_nifty_reg_f3d = join(nifti_directory, 'reg_f3d')
        path_to_nifty_reg_resample = join(nifti_directory, 'reg_resample')
        return cls(path_to_nifty_reg_aladin=path_to_nifty_reg_aladin,
                  path_to_nifty_reg_f3d=path_to_nifty_reg_f3d,
                  path_to_nifty_reg_resample=path_to_nifty_reg_resample)