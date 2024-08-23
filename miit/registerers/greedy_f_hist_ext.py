from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional, List, Tuple

import greedyfhist, greedyfhist as gfh
from greedyfhist.options import RegistrationOptions
from greedyfhist.registration.greedy_f_hist import RegistrationResult, GroupwiseRegResult
import numpy

from .base_registerer import Registerer, RegistrationResult

@dataclass
class GreedyFHistRegistrationResult(RegistrationResult):

    registration_result: RegistrationResult


@dataclass
class GreedyFHistGroupRegistrationResult(RegistrationResult):
    
    registration_result: GroupwiseRegResult

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
                        moving_img_mask: Optional[numpy.array] = None,
                        fixed_img_mask: Optional[numpy.array] = None,
                        options: Optional[RegistrationOptions] = None
                        ) -> GreedyFHistRegistrationResult:
        if options is None:
            options = RegistrationOptions()
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
        transformed_pointset = self.registerer.transform_pointset(pointset, transformation.registration_result.registration_transforms.backward_transform)
        # transformation_result = self.registerer.transform_pointset(pointset, transformation.backward_displacement_field, **kwargs)
        return transformed_pointset 
    
    def transform_image(self, 
                        image: numpy.array, 
                        transformation: GreedyFHistRegistrationResult, 
                        interpolation_mode: str, 
                        **kwargs: Dict) -> numpy.array:
        warped_image = self.registerer.transform_image(image, transformation.registration_result.registration_transforms.forward_transform, interpolation_mode)
        return warped_image

    @classmethod
    def load_from_config(cls, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}
        registerer = gfh.registration.GreedyFHist.load_from_config(config)
        return cls(registerer=registerer)

    def groupwise_registration(self,
                               image_with_mask_list: List[Tuple[numpy.array, Optional[numpy.array]]],
                               options: Optional[RegistrationOptions] = None) -> Tuple[List[GreedyFHistRegistrationResult], Optional[GreedyFHistGroupRegistrationResult]]:
        if options is None:
            options = RegistrationOptions()
        group_reg, _ = self.registerer.groupwise_registration(image_with_mask_list, options=options)
        # To make it possible to apply transformations separately, we need to split them up before returning.
        transform_list = []
        for idx in range(len(image_with_mask_list)-1):
            transform = group_reg.get_transforms(idx)
            transform_list.append(GreedyFHistRegistrationResult(transform))
        g_reg = GreedyFHistGroupRegistrationResult(group_reg)
        return transform_list, g_reg

        
