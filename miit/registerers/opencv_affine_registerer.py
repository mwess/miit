from dataclasses import dataclass
from typing import Any

import cv2
import numpy, numpy as np
import pandas as pd
from skimage import transform

from .base_registerer import Registerer

from miit.utils.utils import simpleitk_to_skimage_interpolation

@dataclass
class OpenCVAffineTransformation:
    
    transformation_matrix: numpy.ndarray
    height: int
    width: int
    src_pts: numpy.ndarray | None
    dst_pts: numpy.ndarray | None
    

def get_detector(detector_name: str, n_features: int = 0) -> cv2.SIFT | cv2.ORB:
    if detector_name == 'sift':
        return cv2.SIFT_create(n_features)
    elif detector_name == 'orb':
        return cv2.ORB_create(n_features)
    else:
        raise Exception(f'Detector unknown: {detector_name}')


class OpenCVAffineRegisterer(Registerer):
    """
    Landmark registerer for rigid/affine alignments. Based on OpenCV 
    functionality (https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html). 
    This registerer is mostly suitable for matching the
    same tissue section, e.g. when part of the image is cropped and
    needs to be aligned to the reference image. 
    """

    name = 'OpenCVAffineRegisterer'
    
    # TODO: What is the datatype of the return registration
    def register_images(self, 
                        moving_img: numpy.ndarray, 
                        fixed_img: numpy.ndarray, 
                        feature_detector: str = 'sift',
                        rigid: bool = True,
                        rotation: bool = True,
                        warn_angle_deg: float = 1,
                        min_match_count: int = 10,
                        flann_index_kdtree: int = 0,
                        flann_trees: int = 5,
                        flann_checks: int = 50,
                        matching_method: int = cv2.RANSAC,
                        matching_criteron_distance_factor: float = 0.75,
                        n_features: int = 1000,                        
                        verbose: bool = False,
                        **kwargs: dict) -> 'OpenCVAffineTransformation':
        return self.register_(
            moving_img,
            fixed_img,
            feature_detector=feature_detector,
            rigid=rigid,
            rotation=rotation,
            warn_angle_deg=warn_angle_deg,
            min_match_count=min_match_count,
            flann_index_kdtree=flann_index_kdtree,
            flann_trees=flann_trees,
            flann_checks=flann_checks,
            matching_method=matching_method,
            matching_criteron_distance_factor=matching_criteron_distance_factor,
            n_features=n_features,
            verbose=verbose
            )
        
    def register_(self, 
                  moving_img: numpy.ndarray, 
                  fixed_img: numpy.ndarray, 
                  feature_detector: str = 'sift',
                  rigid: bool = True, 
                  rotation: bool = True, 
                  warn_angle_deg: int = 1, 
                  min_match_count: int = 10,
                  flann_index_kdtree: int = 0, 
                  flann_trees: int = 5, 
                  flann_checks: int = 50, 
                  matching_method: int = cv2.RANSAC, 
                  matching_criteron_distance_factor: float = 0.75,
                  n_features: int = 1000,
                  verbose: bool = False) -> 'OpenCVAffineTransformation':
        """
        co-registers two images and returns the moving image warped to fit target_img and the respective transform matrix
        Script is very close to OpenCV2 image co-registration tutorial:
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
        only addition/change here: rigid transform & no rotation option
        
        Args:
            moving_img (numpy.ndarray):
            fixed_img (numpy.ndarray): 
            feature_detector (str):
            rigid (bool): if true only tranlation, rotation, and uniform scale
            rotation (bool): if false no rotation
            warn_angle_deg (int): cuttoff for warning check if supposed rotation angle bigger in case of rotation=False
            min_match_count (int): min good feature matches
            flann_index_kdtree (int): define algorithm for Fast Library for Approximate Nearest Neighbors - see FLANN doc
            flann_trees (int):
            flan_checks (int):
            matching_method (int):
            matching_criteron_distance_factor (float):
            n_features (int):
            verbose (bool):
            
        Returns:
            numpy.ndarray
        """
        all_feature_detectors = ['sift', 'orb']
        if feature_detector not in all_feature_detectors:
            raise Exception(f'Uknown feature detector provided: {feature_detector}')

        if len(fixed_img.shape) > 2:
            fixed_img = cv2.cvtColor(fixed_img, cv2.COLOR_BGR2GRAY)
        if len(moving_img.shape) > 2:
            moving_img = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)
        height, width = fixed_img.shape

        # find the keypoints and descriptors with SIFT
        feature_detector_ = get_detector(feature_detector, n_features=n_features)
        kp1, des1 = feature_detector_.detectAndCompute(moving_img, None)
        kp2, des2 = feature_detector_.detectAndCompute(fixed_img, None)

        index_params = dict(algorithm=flann_index_kdtree, trees=flann_trees)
        search_params = dict(checks=flann_checks)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < matching_criteron_distance_factor* n.distance:
                good.append(m)
        if len(good) > min_match_count:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            if rigid:
                transformation_matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=matching_method,
                                                                ransacReprojThreshold=5.0)
                transformation_matrix = np.vstack([transformation_matrix, [0, 0, 1]])
                if not rotation:
                    angle = np.arcsin(transformation_matrix[0, 1])
                    # TODO: Replace with logger
                    if verbose:
                        print('Current rotation {} degrees'.format(np.rad2deg(angle)))
                    if abs(np.rad2deg(angle)) > warn_angle_deg:
                        # TODO: Replace with logger.
                        if verbose:
                            print('Warning: calculated rotation > {} degrees!'.format(warn_angle_deg))
                    pure_scale = transformation_matrix[0, 0] / np.cos(angle)
                    transformation_matrix[0, 0] = pure_scale
                    transformation_matrix[0, 1] = 0
                    transformation_matrix[1, 0] = 0
                    transformation_matrix[1, 1] = pure_scale
            else:
                transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, matching_method, 3.0)
        else:
            # TODO: Replace with logger
            if verbose:
                print("Not enough matches are found - {}/{}".format(len(good), min_match_count))
            transformation_matrix = None
            src_pts = None
            dst_pts = None

        return OpenCVAffineTransformation(transformation_matrix, 
                                          height, width, 
                                          src_pts=src_pts, 
                                          dst_pts=dst_pts)        

    def transform_pointset(self, 
                           pointset: numpy.ndarray, 
                           transformation: OpenCVAffineTransformation, 
                           **kwargs: dict) -> numpy.ndarray:
        transformed_pointset = (transformation.transformation_matrix @ np.hstack((pointset, np.ones((pointset.shape[0], 1)))).T).T
        transformed_pointset = transformed_pointset[:,:2]
        return transformed_pointset

    def transform_image(self, 
                        image: numpy.ndarray, 
                        transformation: OpenCVAffineTransformation, 
                        interpolation_mode: int | str, 
                        **kwargs: dict) -> numpy.ndarray:
        order = simpleitk_to_skimage_interpolation(interpolation_mode)
        tform = transform.AffineTransform(transformation.transformation_matrix)
        transformed_image = transform.warp(image, tform.inverse, output_shape=(transformation.height, transformation.width), order=order, cval=0)
        return transformed_image