from dataclasses import dataclass
import os
from os.path import join, exists
import shlex
import shutil
import subprocess
from typing import Any, Dict, Optional, List

import cv2
import numpy
import numpy as np
import SimpleITK as sitk

from miit.registerers.base_registerer import Registerer, RegistrationResult


def resize_image_simple_sitk(image, res, out_type=np.float32):
    img_np = sitk.GetArrayFromImage(image)
    new_img_np = cv2.resize(img_np.astype(np.float32), (res[1], res[0]), 0, 0, interpolation=cv2.INTER_NEAREST)
    return sitk.GetImageFromArray(new_img_np.astype(np.float32))


@dataclass
class NiftiRegRegistrationResult(RegistrationResult):
    rigid_transform_path: str
    cmdln_returns: List[Any]
    pass


@dataclass
class NiftyRegWrapper(Registerer):

    name = 'NiftyRegWrapper'
    path_to_nifty_reg_aladin = '/home/maximilw/workbench/applications/niftyreg/build/reg-apps/reg_aladin'
    path_to_nifty_reg_f3d = '/home/maximilw/workbench/applications/niftyreg/build/reg-apps/reg_f3d'
    path_to_nifty_resample = '/home/maximilw/workbench/applications/niftyreg/build/reg-apps/reg_resample'
    
    def register_images(self, 
                        moving_img: numpy.array, 
                        fixed_img: numpy.array, 
                        output_dir: Optional[str] = None,
                        use_half_res: bool = True,
                        reg_mode: str = 'rigid',
                        **kwargs: Dict) -> RegistrationResult:
    # Export images
        if use_half_res:
            fixed_half_size = [x//2 for x in sitk.GetArrayFromImage(fixed_img).shape]
            fixed_image_half_res = resize_image_simple_sitk(fixed_img, fixed_half_size) 
            moving_half_size = [x//2 for x in sitk.GetArrayFromImage(moving_img).shape]
            moving_image_half_res = resize_image_simple_sitk(moving_img, moving_half_size)
        else:
            fixed_image_half_res = fixed_img
            moving_image_half_res = moving_img
        # Write to file
        cmd_returns = []
        if not exists(output_dir):
            os.mkdir(output_dir)
        fixed_full_path = join(output_dir, 'fixed_full.nii.gz')
        moving_full_path = join(output_dir, 'moving_full.nii.gz')
        fixed_path = join(output_dir, 'fixed.nii.gz')
        moving_path = join(output_dir, 'moving.nii.gz')
        warped_rig_image_path = join(output_dir, 'warped_rig_image.nii.gz')
        rigid_transform_path = join(output_dir, 'Rigid.txt')
        sitk.WriteImage(fixed_image_half_res, fixed_path)
        sitk.WriteImage(moving_image_half_res, moving_path)
        sitk.WriteImage(fixed_img, fixed_full_path)
        sitk.WriteImage(moving_img, moving_full_path)
    
        # Perform registrations
        cmd_aladin_rig = f"""'{self.path_to_nifty_reg_aladin}' -ref '{fixed_path}' -flo '{moving_path}' -res '{warped_rig_image_path}' -aff '{rigid_transform_path}' -rigOnly"""
        ret_aladin_rig = subprocess.run(shlex.split(cmd_aladin_rig), capture_output=True)
        if len(ret_aladin_rig.stderr) != 0:
            print('Error during rigid registration:')
            print(ret_aladin_rig.stderr.decode('utf-8'))
        cmd_returns.append(ret_aladin_rig)    
        # Load resampled 
        warped_rig_image = sitk.ReadImage(warped_rig_image_path)
        if use_half_res:
            target_img_size = sitk.GetArrayFromImage(moving_img).shape
            warped_rig_image = resize_image_simple_sitk(warped_rig_image, target_img_size)
        if exists(output_dir):
            shutil.rmtree(output_dir)
        return (warped_rig_image, 
                rigid_transform_path,
                fixed_path,
                cmd_returns)
    
    def transform_image(self, image: numpy.array, transformation: RegistrationResult, interpolation_mode: str, **kwargs: Dict) -> numpy.array:
        return super().transform_image(image, transformation, interpolation_mode, **kwargs)
    
    def transform_pointset(self, pointset: numpy.array, transformation: RegistrationResult, **kwargs: Dict) -> numpy.array:
        return super().transform_pointset(pointset, transformation, **kwargs)
    
    @classmethod
    def load_from_config(cls, config: Dict[str, Any]) -> Registerer:
        
        path_to_nifty_reg_aladin = config['path_to_nifty_reg_aladin']
        path_to_nifty_reg_f3d = config['path_to_nifty_reg_f3d']
        path_to_nifty_resample = config['path_to_nifty_resample']
        
        return cls(path_to_nifty_reg_aladin=path_to_nifty_reg_aladin,
                  path_to_nifty_reg_f3d=path_to_nifty_reg_f3d,
                  path_to_nifty_resample=path_to_nifty_resample)