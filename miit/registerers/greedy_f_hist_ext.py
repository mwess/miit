from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, List, Tuple

import greedyfhist
import greedyfhist as gfh
from greedyfhist.options import Options, GreedyOptions
import numpy
import SimpleITK

from .base_registerer import Registerer, RegistrationResult

@dataclass
class GreedyFHistRegistrationResult(RegistrationResult):

    registration_result: greedyfhist.registration.greedy_f_hist.RegistrationResult

@dataclass
class GreedyFHistGroupRegistrationResult(RegistrationResult):
    
    registration_result: greedyfhist.registration.greedy_f_hist.GroupwiseRegResult
    

@dataclass
class GreedyFHistExt(Registerer):
    """
    Wrapper for GreedyFHist algorithm.
    """
    name: ClassVar[str] = 'GreedyFHist'
    registerer: greedyfhist.registration.greedy_f_hist.GreedyFHist
    
    # TODO: What is the datatype of the return registration
    def register_images(self, 
                        moving_img: numpy.array, 
                        fixed_img: numpy.array, 
                        **kwargs: Dict) -> GreedyFHistRegistrationResult:
        moving_img_mask = kwargs.get('moving_img_mask', None)
        fixed_img_mask = kwargs.get('fixed_img_mask', None)
        options = Options()
        options.parse_dict(kwargs)
        reg_result = self.registerer.register(moving_img=moving_img,
                                              fixed_img=fixed_img,
                                              moving_img_mask=moving_img_mask,
                                              fixed_img_mask=fixed_img_mask,
                                              options=options)
        greedy_result = GreedyFHistRegistrationResult(reg_result)
        return greedy_result

    def transform_pointset(self, 
                           pointset: numpy.array, 
                           transformation: GreedyFHistRegistrationResult, 
                           **kwargs: Dict) -> numpy.array:
        transformed_pointset = self.registerer.transform_pointset(pointset, transformation.registration_result.moving_transform)
        # transformation_result = self.registerer.transform_pointset(pointset, transformation.backward_displacement_field, **kwargs)
        return transformed_pointset 
    
    def transform_image(self, 
                        image: numpy.array, 
                        transformation: GreedyFHistRegistrationResult, 
                        interpolation_mode: str, 
                        **kwargs: Dict) -> numpy.array:
        warped_image = self.registerer.transform_image(image, transformation.registration_result.fixed_transform, interpolation_mode)
        return warped_image
        # transformation_result = self.registerer.transform_image(image, transformation.forward_displacement_field, interpolation_mode, **kwargs)
        # return transformation_result.final_transform.registered_image

    # TODO: Remove this.
    def get_default_args(self):
        return gfh.registration.get_default_args()

    @classmethod
    def load_from_config(cls, config: Optional[Dict[str, Any]]):
        if config is None:
            config = {}
        registerer = gfh.registration.GreedyFHist.load_from_config(config)
        return cls(registerer=registerer)

    def groupwise_registration(self,
                               image_with_mask_list: List[Tuple[numpy.array, Optional[numpy.array]]],
                               skip_deformable_registration=False,
                               apply_transforms: bool = True,
                               **kwargs: Dict) -> Tuple[List[GreedyFHistRegistrationResult], Optional[GreedyFHistGroupRegistrationResult]]:
        group_reg, _ = self.registerer.groupwise_registration(image_with_mask_list, skip_deformable_registration=skip_deformable_registration)
        # To make it possible to apply transformations separately, we need to split them up before returning.
        transform_list = []
        for idx in range(len(image_with_mask_list)-1):
            transform = group_reg.get_transforms(idx)
            transform_list.append(GreedyFHistRegistrationResult(transform))
        g_reg = GreedyFHistGroupRegistrationResult(group_reg)
        return transform_list, g_reg

        
