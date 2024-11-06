from dataclasses import dataclass
from typing import ClassVar, Callable

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
                        moving_img: numpy.ndarray, 
                        fixed_img: numpy.ndarray, 
                        moving_img_mask: numpy.ndarray | None = None,
                        fixed_img_mask: numpy.ndarray | None = None,
                        options: RegistrationOptions | None = None) -> GreedyFHistRegistrationResult:
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
                           pointset: numpy.ndarray, 
                           transformation: GreedyFHistRegistrationResult, 
                           do_reverse_transform: bool = False,
                           **kwargs: dict) -> numpy.ndarray:
        reg_transform = transformation.registration_result.registration if not do_reverse_transform else transformation.registration_result.reverse_registration
        transformed_pointset = self.registerer.transform_pointset(pointset, reg_transform.backward_transform)
        return transformed_pointset 
    
    def transform_image(self, 
                        image: numpy.ndarray, 
                        transformation: GreedyFHistRegistrationResult, 
                        interpolation_mode: str, 
                        do_reverse_transform: bool = False,
                        **kwargs: dict) -> numpy.ndarray:
        reg_transform = transformation.registration_result.registration if not do_reverse_transform else transformation.registration_result.reverse_registration
        warped_image = self.registerer.transform_image(image, reg_transform.forward_transform, interpolation_mode)
        return warped_image

    @classmethod
    def init_registerer(cls, 
                        path_to_greedy: str = '',
                        use_docker_container: bool = False,
                        segmentation_function: Callable[[numpy.ndarray], numpy.ndarray] | None = None):
        registerer = gfh.registration.GreedyFHist(path_to_greedy=path_to_greedy,
                                                  use_docker_container=use_docker_container,
                                                  segmentation_function=segmentation_function)
        return cls(registerer=registerer)

    def groupwise_registration(self,
                               image_with_mask_list: list[tuple[numpy.ndarray, numpy.ndarray | None]],
                               options: RegistrationOptions | None = None) -> tuple[list[GreedyFHistRegistrationResult], GreedyFHistGroupRegistrationResult | None]:
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