from dataclasses import dataclass, field
import os
from os.path import join

import cv2
import numpy, numpy as np
import pydicom
import SimpleITK as sitk
import torch, torch.nn as nn
from torch.autograd import Variable


from .model.ProsRegNet_model import ProsRegNet
from .geotnf.transformation import GeometricTnf
from .geotnf.transformation_high_res import GeometricTnf_high_res
from .geotnf.point_tnf import PointTnf
from .utils import scale_image, get_padding_params, normalize_image

from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.distance_unit import DUnit


def preprocess_image(image):
    resizeCNN = GeometricTnf(out_h=240, out_w=240, use_cuda = False) 

    # convert to torch Variable
    image = np.expand_dims(image.transpose((2,0,1)),0)
    image = torch.Tensor(image.astype(np.float32)/255.0)
    image_var = Variable(image,requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resizeCNN(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var


def runCnn(model_cache: tuple[ProsRegNet, ProsRegNet, bool, bool, bool], 
            source_image: np.ndarray, 
            target_image: np.ndarray, 
            source_mask: np.ndarray | None = None,
            target_mask: np.ndarray | None = None, 
            use_mask_for_affine: bool = True,
            out_high_res_half_size: int = 128):
    model_aff, model_tps, do_aff, do_tps, use_cuda = model_cache
    
    tpsTnf = GeometricTnf(geometric_model='tps', use_cuda=use_cuda)
    affTnf = GeometricTnf(geometric_model='affine', use_cuda=use_cuda)
    
    tpsTnf_high_res = GeometricTnf_high_res(geometric_model='tps', use_cuda=use_cuda)
    affTnf_high_res = GeometricTnf_high_res(geometric_model='affine', use_cuda=use_cuda)
    
    # copy MRI image to 3 channels 
    target_image3d = np.zeros((target_image.shape[0], target_image.shape[1], 3), dtype=int)
    target_image3d[:, :, 0] = target_image
    target_image3d[:, :, 1] = target_image
    target_image3d[:, :, 2] = target_image
    target_image = np.copy(target_image3d)

    #### begin new code, affine registration using the masks only
    source_image_mask = np.copy(source_image)
    source_image_mask[np.any(source_image_mask > 5, axis=-1)] = 255
    target_image_mask = np.copy(target_image)
    target_image_mask[np.any(target_image_mask > 5, axis=-1)] = 255
    source_image_mask_var = preprocess_image(source_image_mask)
    target_image_mask_var = preprocess_image(target_image_mask)
    
    if use_cuda:
        source_image_mask_var = source_image_mask_var.cuda()
        target_image_mask_var = target_image_mask_var.cuda()
    batch_mask = {'source_image': source_image_mask_var, 'target_image':target_image_mask_var}
    #### end new code  

    source_image_var = preprocess_image(source_image)
    target_image_var = preprocess_image(target_image)
    
    
    source_image_var_high_res = preprocess_image_high_res(source_image)
    target_image_var_high_res = preprocess_image_high_res(target_image)
    
    if use_cuda:
        source_image_var = source_image_var.cuda()
        target_image_var = target_image_var.cuda()
        source_image_var_high_res = source_image_var_high_res.cuda()
        target_image_var_high_res = target_image_var_high_res.cuda()

    batch = {'source_image': source_image_var, 'target_image':target_image_var}
    batch_high_res = {'source_image': source_image_var_high_res, 'target_image':target_image_var_high_res}

    if do_aff:
        model_aff.eval()
    if do_tps:
        model_tps.eval()

    # Evaluate models
    thetas = {}
    thetas['affine'] = []
    thetas['tps'] = []
    if do_aff:
        #theta_aff=model_aff(batch)
        #### affine registration using the masks only
        if use_mask_for_affine:
            theta_aff=model_aff(batch_mask)
        else:
            theta_aff = model_aff(batch)
        thetas['affine'].append(theta_aff)
        warped_image_aff_high_res = affTnf_high_res(batch_high_res['source_image'], theta_aff.view(-1,2,3))
        warped_image_aff = affTnf(batch['source_image'], theta_aff.view(-1,2,3))
        
        ###>>>>>>>>>>>> do affine registration one more time<<<<<<<<<<<<
        warped_mask_aff = affTnf(source_image_mask_var, theta_aff.view(-1,2,3))
        theta_aff=model_aff({'source_image': warped_mask_aff, 'target_image': target_image_mask_var})
        warped_image_aff_high_res = affTnf_high_res(warped_image_aff_high_res, theta_aff.view(-1,2,3))
        warped_image_aff = affTnf(warped_image_aff, theta_aff.view(-1,2,3))
        thetas['affine'].append(theta_aff)
        ###>>>>>>>>>>>> do affine registration one more time<<<<<<<<<<<<

    if do_aff and do_tps:
        theta_aff_tps=model_tps({'source_image': warped_image_aff, 'target_image': batch['target_image']})   
        warped_image_aff_tps_high_res = tpsTnf_high_res(warped_image_aff_high_res,theta_aff_tps)
        thetas['tps'].append(theta_aff_tps)

    # Un-normalize images and convert to numpy
    if do_aff:
        warped_image_aff_np_high_res = normalize_image(warped_image_aff_high_res,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()

    if do_aff and do_tps:
        warped_image_aff_tps_np_high_res = normalize_image(warped_image_aff_tps_high_res,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()      
    
    warped_image_aff_np_high_res[warped_image_aff_np_high_res < 0] = 0
    warped_image_aff_tps_np_high_res[warped_image_aff_tps_np_high_res < 0] = 0    
    
    theta = theta_aff_tps if do_tps else theta_aff
    return warped_image_aff_tps_np_high_res, thetas


def preprocess_hist(img: numpy.ndarray, 
                         mask: numpy.ndarray,
                         padding: int = 30) -> tuple[numpy.ndarray, numpy.ndarray, dict]:
    """Does the following preprocessing steps:
        1. Applies mask.
        2. Crops around the masked area.
        3. Adds padding.
    """

    # Downsize by 4. Or resize to 500px
    if len(mask.shape) == 2:
        mask_ = np.expand_dims(mask, -1)
    else:
        mask_ = mask
    img = img * mask_
    # This doesnt make sense, since the cropping happens immediately afterwards.
    # Maybe a refactoring mistake?

    # create a bounding box around slice
    preprocessing_steps = {}
    points = np.argwhere(mask != 0)
    points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
    y, x, h, w = cv2.boundingRect(points) # create a rectangle around those points
    img = img[x:x+w, y:y+h]
    mask = mask[x:x+w, y:y+h]
    preprocessing_steps['crop'] = {}
    preprocessing_steps['crop']['y'] = y
    preprocessing_steps['crop']['x'] = x
    preprocessing_steps['crop']['h'] = h
    preprocessing_steps['crop']['w'] = w
    # Pad to a square shape now.
    max_size = max(img.shape[0], img.shape[1])
    ylo, yhi, xlo, xhi = get_padding_params(img, max_size)
    ylo += padding
    yhi += padding
    xlo += padding
    xhi += padding
    preprocessing_steps['sym_pad'] = {}
    preprocessing_steps['sym_pad']['ylo'] = ylo
    preprocessing_steps['sym_pad']['yhi'] = yhi
    preprocessing_steps['sym_pad']['xlo'] = xlo
    preprocessing_steps['sym_pad']['xhi'] = xhi
    img = np.pad(img,((xlo, xhi),(ylo, yhi),(0,0)),'constant', constant_values=0)
    mask = np.pad(mask,((xlo, xhi), (ylo, yhi)),'constant', constant_values=0)
    return img, mask, preprocessing_steps


def preprocess_mri(mri: numpy.ndarray,
                    mri_masks: numpy.ndarray,
                    padding: int = 30):
    if len(mri.shape) == 0:
        mri = np.expand_dims(mri, 0)
    prepr_mri = []

    # Make a first pass to get the maximum size
    max_size = 0
    for i in range(mri_masks.shape[0]):
        mask = mri_masks[i]
        points = np.argwhere(mask != 0)
        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
        y, x, h, w = cv2.boundingRect(points) # create a rectangle around those points
        max_size_ = max(w, h)
        if max_size_ > max_size:
            max_size = max_size_

    max_size = max_size + 2 * padding
    for i in range(mri.shape[0]):
        prepr_steps = {}
        mri_slice = mri[i]
        mri_mask = mri_masks[i]

        if np.sum(mri_mask) == 0:
            continue

        mri_slice = mri_slice * mri_mask
        mri_mask = mri_mask * 255
        points = np.argwhere(mri_mask != 0)
        points = np.fliplr(points) # store them in x,y coordinates instead of row,col indices
        y, x, h, w = cv2.boundingRect(points) # create a rectangle around those points
        prepr_steps['crop'] = {}
        prepr_steps['crop']['y'] = y
        prepr_steps['crop']['x'] = x
        prepr_steps['crop']['h'] = h
        prepr_steps['crop']['w'] = w
        prepr_steps['crop']['size_before_crop'] = mri_slice.shape
        prepr_steps['slice'] = i
        prepr_steps['crop']['x_right'] = mri_slice.shape[0] - w - x
        prepr_steps['crop']['y_right'] = mri_slice.shape[1] - h - y

        mri_slice = mri_slice[x:x+w, y:y+h]
        mri_slice = mri_slice / max(int(np.max(mri_slice) / 255), 1)
        mri_slice = mri_slice * 25.5/(np.max(mri_slice)/10)
        
        ylo, yhi, xlo, xhi = get_padding_params(mri_slice, max_size)
        prepr_steps['sym_pad'] = {}
        prepr_steps['sym_pad']['ylo'] = ylo
        prepr_steps['sym_pad']['yhi'] = yhi
        prepr_steps['sym_pad']['xlo'] = xlo
        prepr_steps['sym_pad']['xhi'] = xhi

        mri_slice = np.pad(mri_slice,((xlo, xhi),(ylo, yhi)),'constant', constant_values=0)
        mri_mask = np.pad(mri_mask,((xlo, xhi), (ylo, yhi)),'constant', constant_values=0)
             
        prepr_mri.append((i, mri_slice, mri_mask, prepr_steps))
    return prepr_mri


def preprocess_image_high_res(image, half_out_size=500):
    resizeCNN = GeometricTnf(out_h=half_out_size*2, out_w=half_out_size*2, use_cuda = False) 

    # convert to torch Variable
    image = np.expand_dims(image.transpose((2,0,1)),0)
    image = torch.Tensor(image.astype(np.float32)/255.0)
    image_var = Variable(image,requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image_var = resizeCNN(image_var)

    # Normalize image
    image_var = normalize_image(image_var)

    return image_var


@dataclass
class ProsRegNetTransformation:

    transform: dict


@dataclass(kw_only=True)
class ProsRegNetExt:
    """Wrapper class around the ProsRegNet registration algorithm. If you use this code, please cite the authors:
    
    Shao, Wei, et al. "ProsRegNet: A Deep Learning Framework for Registration of MRI and Histopathology Images of the Prostate."""
    

    use_cuda: bool
    affine_model: nn.Module | None = None
    tps_model: nn.Module | None = None
    

    @classmethod
    def init_registerer(cls, 
                        path_to_affine_model: str | None = None,
                        path_to_tps_model: str | None = None,
                        use_cuda: bool | None = None,
                        feature_extraction_cnn: str = 'resnet101'):
        if use_cuda is None:
            use_cuda = torch.cuda.is_available()
        if path_to_affine_model:
            affine_model = ProsRegNet(use_cuda=use_cuda, geometric_model='affine', feature_extraction_cnn=feature_extraction_cnn)
        if path_to_tps_model:
            tps_model = ProsRegNet(use_cuda=use_cuda, geometric_model='tps', feature_extraction_cnn=feature_extraction_cnn) 
        return cls(affine_model=affine_model,
                   tps_model=tps_model,
                   use_cuda=use_cuda)   

    def register_hist_mri_stack(self, 
                 hist_img: numpy.ndarray, 
                 mri: numpy.ndarray,
                 hist_mask: numpy.ndarray | None = None,
                 mri_mask: numpy.ndarray | None = None,
                 hist_resolution: DUnit | tuple[DUnit, DUnit] | None = None,
                 mri_resolution: DUnit | tuple[DUnit, DUnit] | None = None,
                 reg_resolution: DUnit | tuple[DUnit, DUnit] | None = None,
                 do_affine: bool = True,
                 do_tps: bool = True,
                 use_masks_as_affine: bool = True,
                 use_cuda: bool | None = None,
                 **kwargs: dict) -> ProsRegNetTransformation:
        if hist_resolution is None:
            hist_resolution = DUnit.default_dunit()
        if isinstance(hist_resolution, DUnit):
            hist_resolution = (hist_resolution, hist_resolution)
        
        if mri_resolution is None:
            mri_resolution = DUnit.default_dunit()
        if isinstance(mri_resolution, DUnit):
            mri_resolution = (mri_resolution, mri_resolution)
        
        if reg_resolution is None:
            reg_resolution = DUnit('75', 'um')
        if isinstance(reg_resolution, DUnit):
            reg_resolution = (reg_resolution, reg_resolution)

        transform_steps = {
            'hist': {},
            'mri': {}
        }

        transform_steps['hist']['hist_input_size'] = hist_img.shape[:2]
        transform_steps['hist']['hist_resolution'] = hist_resolution
        transform_steps['mri']['mri_input_size'] = mri.shape[1:]
        transform_steps['mri']['mri_resolution'] = mri_resolution
        transform_steps['reg_resolution'] = reg_resolution

        hist_conversion_factor_w = reg_resolution[0].get_conversion_factor(hist_resolution[0])     
        hist_conversion_factor_h = reg_resolution[1].get_conversion_factor(hist_resolution[1])     
        
        mri_conversion_factor_w = reg_resolution[0].get_conversion_factor(mri_resolution[0])
        mri_conversion_factor_h = reg_resolution[1].get_conversion_factor(mri_resolution[1])
        
        transform_steps['hist']['conversion_factors'] = (hist_conversion_factor_w, hist_conversion_factor_h)
        transform_steps['mri']['conversion_factors'] = (mri_conversion_factor_w, mri_conversion_factor_h)

        hist_img = scale_image(hist_img, (hist_conversion_factor_w, hist_conversion_factor_h), interpolation=cv2.INTER_CUBIC)
        hist_mask = scale_image(hist_mask, (hist_conversion_factor_w, hist_conversion_factor_h), interpolation=cv2.INTER_NEAREST)

        hist_img, hist_mask, hist_prepr = preprocess_hist(hist_img, hist_mask)
        transform_steps['hist']['hist_prepr'] = hist_prepr
        transform_steps['hist']['size_after_preprocessing'] = hist_img.shape[:2]

        mri_swapped = np.moveaxis(mri, 0, 2)
        mri_mask_swapped = np.moveaxis(mri_mask, 0, 2)
        
        mri_swapped = scale_image(mri_swapped, (mri_conversion_factor_w, mri_conversion_factor_h), interpolation=cv2.INTER_CUBIC)                
        mri_mask_swapped = scale_image(mri_mask_swapped, (mri_conversion_factor_w, mri_conversion_factor_h), interpolation=cv2.INTER_NEAREST)
        
        mri_swapped = np.moveaxis(mri_swapped, 2, 0)
        mri_mask_swapped = np.moveaxis(mri_mask_swapped, 2, 0)

        mri_prep = preprocess_mri(mri_swapped, mri_mask_swapped)

        mri_slices = {}
        mri_masks = {}
        mri_prep_steps = {}
        for (slice_idx, mri_slice, mri_mask, prep_step) in mri_prep:
            mri_slices[slice_idx] = mri_slice
            mri_masks[slice_idx] = mri_mask
            mri_prep_steps[slice_idx] = prep_step

        transform_steps['mri']['mri_prepr'] = mri_prep_steps
        transform_steps['mri']['size_before_transform'] = mri_prep[0][1].shape
        transform_steps['reg_opts'] = (do_affine, do_tps)

        reg_result = self.register_(
            mri_slices,
            hist_img,
            mri_masks,
            do_affine,
            do_tps,
            use_masks_as_affine,
            use_cuda 
        )
        transform_steps['mri']['reg_result'] = reg_result
        prosregnet_transform = ProsRegNetTransformation(transform=transform_steps)
        return prosregnet_transform

    def register_(self, 
                  mri_slices: dict[int, numpy.ndarray], 
                  hist_image: numpy.ndarray,                   
                  mri_masks: dict[int, numpy.ndarray] | None = None,
                  hist_mask: numpy.ndarray | None = None,
                  do_affine: bool = True,
                  do_tps: bool = True,
                  use_masks_as_affine: bool = True,
                  geometric_out_half_size: int = 128,
                  use_cuda: bool | None = None,
                  verbose: bool = False): 
        ####### grab files that were preprocessed     
        if use_cuda is None:
            use_cuda = self.use_cuda

        reg_data = {}
        model_cache = (self.affine_model, self.tps_model, do_affine, do_tps, use_cuda)
        if verbose:
            print(f'Number of mri slices left: {len(mri_slices)}')      
        for idx, key in enumerate(mri_slices): 
            if verbose:
                print(idx)
            reg_data_ = {}
            mri_slice = mri_slices[key]
            mri_mask = None if mri_masks is None else mri_masks[key]
            w_new, h_new = mri_slice.shape[:2]
            affTps, thetas = runCnn(model_cache=model_cache, 
                                    source_image=hist_image, 
                                    target_image=mri_slice, 
                                    source_mask=hist_mask,
                                    target_mask=mri_mask,
                                    use_mask_for_affine=use_masks_as_affine,
                                    out_high_res_half_size=geometric_out_half_size,
                                    )        
            size_after_transform = affTps.shape[:2]
            affTps = cv2.resize(affTps*255, (int(h_new),  int(w_new)), interpolation=cv2.INTER_CUBIC)   
            
            reg_data_['thetas'] = thetas
            reg_data_['affTps'] = affTps
            reg_data_['hist_image'] = hist_image
            reg_data_['mri_slice'] = mri_slice
            reg_data_['size_after_transform'] = size_after_transform
            reg_data_['geometric_out_half_size'] = geometric_out_half_size
            reg_data[key] = reg_data_
        
        return reg_data

    def transform_pointset_to_stack(self,
                           pointset: numpy.ndarray,
                           transformation: ProsRegNetTransformation) -> dict[int, numpy.ndarray]:
        transformation = transformation.transform
        use_cuda = transformation.get('use_cuda', torch.cuda.is_available())
        hist_w, hist_h = transformation['hist']['conversion_factors']
        pointset = pointset.astype(float)
        pointset[:, 0] *= float(hist_w)
        pointset[:, 1] *= float(hist_h)

        # Step 2
        y = transformation['hist']['hist_prepr']['crop']['y']
        x = transformation['hist']['hist_prepr']['crop']['x']

        pointset[:, 0] -= y
        pointset[:, 1] -= x

        ylo = transformation['hist']['hist_prepr']['sym_pad']['ylo']
        xlo = transformation['hist']['hist_prepr']['sym_pad']['xlo']

        pointset[:, 0] += ylo
        pointset[:, 1] += xlo
        point_tnf = PointTnf(use_cuda=use_cuda)
        
        geometric_out_size2 = 1000
        hist_size_after_preprocessing = transformation['hist']['size_after_preprocessing']
        scale_w = geometric_out_size2 / hist_size_after_preprocessing[0]
        scale_h = geometric_out_size2 / hist_size_after_preprocessing[1]

        pointset[:, 0] *= scale_w
        pointset[:, 1] *= scale_h

        warped_pointsets = {}
        for key in transformation['mri']['reg_result']:
            reg_result = transformation['mri']['reg_result'][key]
            thetas = reg_result['thetas']
            size_after_transform = reg_result['size_after_transform']
            geometric_out_half_size = reg_result['geometric_out_half_size']
            geometric_out_size = 2 * geometric_out_half_size
            pointset_ = pointset.copy()
            scale_f = geometric_out_size / geometric_out_size2
            pointset_ *= scale_f
            pointset_ = np.transpose(pointset_)
            pointset_ = np.expand_dims(pointset_, 0)
            
            (do_affine, do_tps) = transformation['reg_opts']

            warped_pointset = torch.from_numpy(pointset_).float()
            if do_affine:
                theta_aff1, theta_aff2 = thetas['affine']
                warped_pointset = point_tnf.affPointTnf(theta_aff1, warped_pointset)
                warped_pointset = point_tnf.affPointTnf(theta_aff2, warped_pointset)
            
            if do_affine and do_tps: 
                theta_aff_tps = thetas['tps'][0]
                warped_pointset = point_tnf.tpsPointTnf(theta_aff_tps, warped_pointset)
            
            warped_pointset = warped_pointset.data.cpu().numpy()
            warped_pointset = np.transpose(warped_pointset.squeeze())
             
            mri_target_size = transformation['mri']['reg_result'][key]['mri_slice'].shape

            scale_w = mri_target_size[0] / size_after_transform[0]
            scale_h = mri_target_size[1] / size_after_transform[1]
            warped_pointset[:, 0] *= scale_w
            warped_pointset[:, 1] *= scale_h 

            xlo = transformation['mri']['mri_prepr'][key]['sym_pad']['xlo']
            ylo = transformation['mri']['mri_prepr'][key]['sym_pad']['ylo']
            
            warped_pointset[:, 0] -= ylo
            warped_pointset[:, 1] -= xlo

            x = transformation['mri']['mri_prepr'][key]['crop']['x']
            y = transformation['mri']['mri_prepr'][key]['crop']['y']

            warped_pointset[:, 0] += y
            warped_pointset[:, 1] += x
            scale_w = 1/float(transformation['mri']['conversion_factors'][0])
            scale_h = 1/float(transformation['mri']['conversion_factors'][1])

            warped_pointset[:, 0] *= scale_w
            warped_pointset[:, 1] *= scale_h
            
            warped_pointsets[key] = warped_pointset.copy()
        
        return warped_pointsets
            

    def transform_image_to_stack(self, 
                        image: numpy.ndarray, 
                        transformation: ProsRegNetTransformation, 
                        interpolation_mode: str = 'LINEAR') -> dict[int, numpy.ndarray]:
        """
        Preprocessing steps to apply:
            1. Apply conversion
            2. Apply hist preprocessing
            3. Transform image
            4. Reverse apply mri preprocessing
            5. Reverse apply mri conversion
        """
        transformation = transformation.transform
        use_cuda = transformation.get('use_cuda', torch.cuda.is_available())
        # Step 1.
        hist_w, hist_h = transformation['hist']['conversion_factors']
        if interpolation_mode == 'NN':
            cv_interpolation = cv2.INTER_NEAREST
        elif interpolation_mode == 'LINEAR':
            cv_interpolation = cv2.INTER_LINEAR
        else:
            cv_interpolation = cv2.INTER_CUBIC

        image = scale_image(image, (hist_w, hist_h), interpolation=cv_interpolation)

        # Step 2.
        transformation['hist']['hist_prepr']
        y = transformation['hist']['hist_prepr']['crop']['y']
        x = transformation['hist']['hist_prepr']['crop']['x']
        h = transformation['hist']['hist_prepr']['crop']['h']
        w = transformation['hist']['hist_prepr']['crop']['w']

        image = image[x:x+w, y:y+h]

        ylo = transformation['hist']['hist_prepr']['sym_pad']['ylo']
        yhi = transformation['hist']['hist_prepr']['sym_pad']['yhi']
        xlo = transformation['hist']['hist_prepr']['sym_pad']['xlo']
        xhi = transformation['hist']['hist_prepr']['sym_pad']['xhi']
        if len(image.shape) == 2:
            image = np.pad(image,((xlo, xhi),(ylo, yhi)),'constant', constant_values=0)
        else:
            image = np.pad(image,((xlo, xhi),(ylo, yhi),(0,0)),'constant', constant_values=0)

        # Step 3
        source_image_var_high_res = preprocess_image_high_res(image)
        if use_cuda:
            source_image_var_high_res = source_image_var_high_res.cuda()
        warped_images = {}
        for key in transformation['mri']['reg_result']:
            reg_result = transformation['mri']['reg_result'][key]
            thetas = reg_result['thetas']
            geometric_out_half_size = reg_result['geometric_out_half_size']
            geometric_out_size = 2 * geometric_out_half_size
            tpsTnf_high_res = GeometricTnf_high_res(geometric_model='tps', 
                                                    out_w=geometric_out_size, 
                                                    out_h=geometric_out_size, 
                                                    use_cuda=use_cuda)
            affTnf_high_res = GeometricTnf_high_res(geometric_model='affine',
                                                    out_w=geometric_out_size,
                                                    out_h=geometric_out_size,
                                                    use_cuda=use_cuda)


            (do_affine, do_tps) = transformation['reg_opts']

            if do_affine:
                theta_aff1, theta_aff2 = thetas['affine']
            
            warped_image = affTnf_high_res(source_image_var_high_res.clone(), theta_aff1.view(-1,2,3))
            warped_image = affTnf_high_res(warped_image, theta_aff2.view(-1,2,3))

            if do_affine and do_tps: 
                theta_aff_tps = thetas['tps'][0]
                warped_image = tpsTnf_high_res(warped_image,theta_aff_tps)

            # Un-normalize images and convert to numpy
            warped_image = normalize_image(warped_image,forward=False).data.squeeze(0).transpose(0,1).transpose(1,2).cpu().numpy()
          
            mri_target_size = transformation['mri']['reg_result'][key]['mri_slice'].shape
            mri_target_size = mri_target_size[::-1]
            warped_image = cv2.resize(warped_image*255, (mri_target_size), interpolation=cv2.INTER_CUBIC)   

            # Step4

            ylo = transformation['mri']['mri_prepr'][key]['sym_pad']['ylo']
            yhi = transformation['mri']['mri_prepr'][key]['sym_pad']['yhi']
            xlo = transformation['mri']['mri_prepr'][key]['sym_pad']['xlo']
            xhi = transformation['mri']['mri_prepr'][key]['sym_pad']['xhi']

            xhi = warped_image.shape[0] - xhi
            yhi = warped_image.shape[1] - yhi

            # Reverse of padding is just cropping.
            warped_image = warped_image[xlo:xhi, ylo:yhi]

            # Reverse of cropping is just padding
            y = transformation['mri']['mri_prepr'][key]['crop']['y']
            x = transformation['mri']['mri_prepr'][key]['crop']['x']
            h = transformation['mri']['mri_prepr'][key]['crop']['h']
            w = transformation['mri']['mri_prepr'][key]['crop']['w']

            x_right = transformation['mri']['mri_prepr'][key]['crop']['x_right']
            y_right = transformation['mri']['mri_prepr'][key]['crop']['y_right']

            if len(warped_image.shape) == 2:
                warped_image = np.pad(warped_image, ((x, x_right), (y, y_right)), 'constant', constant_values=0)
            else:
                warped_image = np.pad(warped_image, ((x, x_right), (y, y_right), (0, 0)), 'constant', constant_values=0)            

            target_outputsize = transformation['mri']['mri_input_size']
            target_outputsize = target_outputsize[::-1]
            warped_image = cv2.resize(warped_image, target_outputsize, interpolation=cv_interpolation)
            warped_images[key] = warped_image.copy()
        return warped_images
