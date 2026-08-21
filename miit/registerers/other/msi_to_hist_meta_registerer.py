from dataclasses import dataclass

import numpy, numpy as np

from ..base_registerer import Registerer, RegistrationResult
from ..nifty_reg import NiftyRegWrapper
from miit.utils.image_utils import remove_padding, pad_asym
from miit.utils.imzml import preprocess_histology


@dataclass
class MSItoHistMetaRegistererResult(RegistrationResult):

    reg_result: RegistrationResult
    processing_dict: dict


@dataclass
class MSItoHistMetaRegisterer(Registerer):
    """Registerer for MSI data with Histology. 
    
    This registerer works as meta registerer. Mostly contains preprocessing methods for msi and histology data.
    An additional registration algorithm needs to be provided.
    """

    name = 'MSItoHistMetaRegisterer'
    registerer: Registerer | None = None

    def __post_init__(self):
        if self.registerer is not None:
            if isinstance(self.registerer, MSItoHistMetaRegisterer):
                raise Exception("MSItoHistMetaRegisterer cannot use itself as registerer.")
        if self.registerer is None:
            self.registerer = NiftyRegWrapper.init_registerer()

    def register_msi_histology(self,
                               msi_image: numpy.ndarray,
                               histology_image: numpy.ndarray,
                               msi_image_mask: numpy.ndarray | None = None,
                               histology_image_mask: numpy.ndarray | None = None,
                               use_histology_as_fixed: bool = True,
                               registerer_args: dict | None = None,
                               **kwargs: dict) -> RegistrationResult:
        fixed_image_mask = histology_image_mask
        moving_image_mask = msi_image_mask
        (fixed_image, 
        moving_image, 
        processing_dict
        ) = preprocess_for_registration(histology_image, 
                                        msi_image, 
                                        histology_image_mask,
                                        msi_image_mask)     
        if not use_histology_as_fixed:
            fixed_image, fixed_image_mask, moving_image, moving_image_mask = moving_image, moving_image_mask, fixed_image, fixed_image_mask
            processing_dict['mov_sym_pad'], processing_dict['fix_sym_pad'] = processing_dict['fix_sym_pad'], processing_dict['mov_sym_pad']
        registration_result = self.register_images(
            moving_img=moving_image,
            fixed_img=fixed_image,
            moving_img_mask=moving_image_mask,
            fixed_img_mask=fixed_image_mask,
            registerer_args=registerer_args,
            **kwargs
        )
        msi_to_hist_meta_registerer_result = MSItoHistMetaRegistererResult(
            reg_result=registration_result,
            processing_dict=processing_dict
        )
        return msi_to_hist_meta_registerer_result
        

    def register_images(self, 
                 moving_img: numpy.ndarray,
                 fixed_img: numpy.ndarray, 
                 moving_img_mask: numpy.ndarray | None = None,
                 fixed_img_mask: numpy.ndarray | None = None,
                 registerer_args: dict | None = None, 
                 **kwargs: dict) -> RegistrationResult:
        # # Preprocessing steps
        # # 1. Add computed padding from process dict
        # # 2. add padding
        # msi_image = moving_img
        # histology_image = fixed_img
        # msi_image_mask = moving_img_mask
        # histology_image_mask = fixed_img_mask
        # if registerer_args is None:
        #     registerer_args = {}
        # (fixed_image, 
        # moving_image, 
        # processing_dict
        # ) = preprocess_for_registration(histology_image, 
        #                                 msi_image, 
        #                                 histology_image_mask,
        #                                 msi_image_mask)            
        return self.registerer.register_images(moving_img, 
                                               fixed_img, 
                                               moving_img_mask=moving_img_mask, 
                                               fixed_img_mask=fixed_img_mask, 
                                               **registerer_args)
        # msi_to_hist_meta_registerer_result = MSItoHistMetaRegistererResult(
        #     reg_result=registration_result,
        #     processing_dict=processing_dict
        # )
        # return msi_to_hist_meta_registerer_result
    

    def transform_image(self,
                        image: numpy.ndarray,
                        transformation: MSItoHistMetaRegistererResult,
                        interpolation_mode: str = 'LINEAR') -> numpy.ndarray:
        image = pad_asym(image, transformation.processing_dict['mov_sym_pad'])
        padding = transformation.processing_dict['second_padding']
        if len(image.shape) == 2:
            image = np.pad(image, ((padding, padding), (padding, padding)))
        else:
            image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
        warped_image = self.registerer.transform_image(image, transformation.reg_result, interpolation_mode)
        warped_image = warped_image[padding:-padding, padding:-padding]
        warped_image = remove_padding(warped_image, transformation.processing_dict['fix_sym_pad'])
        return warped_image
    

    def transform_pointset(self,
                           pointset: numpy.ndarray,
                           transformation: MSItoHistMetaRegistererResult) -> numpy.ndarray:
        # Verify this function.
        left, right, top, bottom = transformation.processing_dict['mov_sym_pad']
        padding = transformation.processing_dict['second_padding']
        pointset[:, 0] = pointset[:, 0] + left + padding
        pointset[:, 1] = pointset[:, 1] + top + padding
        warped_pointset = self.registerer.transform_pointset(pointset, transformation.reg_result)
        left, right, top, bottom = transformation.processing_dict['fix_sym_pad']
        warped_pointset[:, 0] = warped_pointset[:, 0] - padding - left
        warped_pointset[:, 1] = warped_pointset[:, 1] - padding - top
        return warped_pointset



def preprocess_for_registration(hist_img: numpy.ndarray, 
                                msi_img: numpy.ndarray,
                                hist_img_mask: numpy.ndarray | None = None,
                                msi_img_mask: numpy.ndarray | None = None,
                                padding: int =100) -> tuple[numpy.ndarray, numpy.ndarray, dict]:
    if msi_img_mask is not None:
        msi_img = msi_img * msi_img_mask
    hist_img_np, process_dict = preprocess_histology(hist_img, msi_img, hist_img_mask)
    # Add global padding
    # First do asym padding
    hist_img_padding = pad_asym(hist_img_np, process_dict['fix_sym_pad'])
    hist_img_padding = np.pad(hist_img_padding, ((padding, padding),(padding, padding)))
    msi_img_padding = pad_asym(msi_img, process_dict['mov_sym_pad'])
    msi_img_padding = np.pad(msi_img_padding, ((padding, padding),(padding, padding)))
    process_dict['second_padding'] = padding
    return hist_img_padding, msi_img_padding, process_dict