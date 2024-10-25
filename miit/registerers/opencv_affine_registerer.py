from dataclasses import dataclass
from typing import Any

import cv2
import numpy
import numpy as np
import pandas as pd
from skimage import transform

from .base_registerer import Registerer

@dataclass
class OpenCVAffineTransformation:
    
    transformation_matrix: numpy.array
    height: int
    width: int
    

class OpenCVAffineRegisterer(Registerer):
    """
    Landmark registerer for rigid/affine alignments. Based on OpenCV 
    functionality. This registerer is mostly suitable for matching the
    same tissue section, e.g. when part of the image is cropped and
    needs to be aligned to the reference image.
    """

    name = 'OpenCVAffineRegisterer'
    
    # TODO: What is the datatype of the return registration
    def register_images(self, 
                        moving_img: numpy.array, 
                        fixed_img: numpy.array, 
                        rigid: bool = True,
                        rotation: bool = False,
                        warn_angle_deg: float = 1,
                        min_match_count: int = 10,
                        flann_index_kdtree: int = 0,
                        flann_trees: int = 5,
                        flann_checks: int = 50,
                        matching_method: int = cv2.RANSAC,
                        verbose: bool = False,
                        **kwargs: dict) -> Any:
        return self.register_(
            moving_img,
            fixed_img,
            rigid=rigid,
            rotation=rotation,
            warn_angle_deg=warn_angle_deg,
            min_match_count=min_match_count,
            flann_index_kdtree=flann_index_kdtree,
            flann_trees=flann_trees,
            flann_checks=flann_checks,
            matching_method=matching_method
            )
        
    def register_(self, 
                  moving_img: numpy.array, 
                  target_img: numpy.array, 
                  rigid: bool = True, 
                  rotation: bool = False, 
                  warn_angle_deg: int = 1, 
                  min_match_count: int = 10,
                  flann_index_kdtree: int = 0, 
                  flann_trees: int = 5, 
                  flann_checks: int = 50, 
                  matching_method: int = cv2.RANSAC, 
                  verbose: bool = False):
        """
        co-registers two images and returns the moving image warped to fit target_img and the respective transform matrix
        Script is very close to OpenCV2 image co-registration tutorial:
        https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
        only addition/change here: rigid transform & no rotation option
        :param moving_img: image supposed to move
        :param target_img: reference / target image
        :param rigid: if true only tranlation, rotation, and uniform scale
        :param rotation: if false no rotation
        :param warn_angle_deg: cuttoff for warning check if supposed rotation angle bigger in case of rotation=False
        :param min_match_count: min good feature matches
        :param flann_index_kdtree: define algorithm for Fast Library for Approximate Nearest Neighbors - see FLANN doc
        :return: moved/transformed image in target image "space" & transformation matrix
        """
        if len(target_img.shape) > 2:
            target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        if len(moving_img.shape) > 2:
            moving_img = cv2.cvtColor(moving_img, cv2.COLOR_BGR2GRAY)
        height, width = target_img.shape

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(moving_img, None)
        kp2, des2 = sift.detectAndCompute(target_img, None)

        index_params = dict(algorithm=flann_index_kdtree, trees=flann_trees)
        search_params = dict(checks=flann_checks)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
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

        return OpenCVAffineTransformation(transformation_matrix, height, width)        

    def transform_pointset(self, 
                           pointset: numpy.array, 
                           transformation: OpenCVAffineTransformation, 
                           **kwargs: dict) -> numpy.array:
        transformed_pointset = (transformation.transformation_matrix @ np.hstack((pointset, np.ones((pointset.shape[0], 1)))).T).T
        pointset_df = pd.DataFrame(transformed_pointset[:,:2]).rename(columns={0:'x', 1:'y'})
        return pointset_df

    def transform_image(self, 
                        image: numpy.array, 
                        transformation: OpenCVAffineTransformation, 
                        interpolation_mode: str, 
                        **kwargs: dict) -> numpy.array:
        if interpolation_mode == 'NN':
            order = 0
        else:
            order = 1
        tform = transform.AffineTransform(transformation.transformation_matrix)
        transformed_image = transform.warp(image, tform.inverse, output_shape=(transformation.height, transformation.width), order=order, cval=0)
        return transformed_image

    @classmethod
    def load_from_config(cls, config: dict[str, Any]) -> 'Registerer':
        return cls()
    
    @classmethod
    def load_registerer(cls, args: dict[str, Any]) -> 'Registerer':
        return cls()