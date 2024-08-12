import json
import os
import uuid
from dataclasses import dataclass
from os.path import exists, join
from typing import Any, ClassVar, Dict, List, Optional, Tuple


import cv2
import numpy, numpy as np
import SimpleITK as sitk


from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.spatial_data.base_types.base_imaging import BaseImage
from miit.utils.utils import create_if_not_exists


@dataclass(kw_only=True)
class Annotation(BaseImage):
    """
    Annotations consists of a spatially resolved map of discrete 
    classes. Classes can either be scalar of vector valued. It
    is assumed that each annotation has the shape of H x W x C.

    Image transformations applied to annotations should use a 
    nearest neighbor interpolation to not introduce new classes.
    """

    interpolation_mode: ClassVar[str] = 'NN'
    labels: List[str] = None

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        # TODO: Add check for image bounds
        if len(self.data.shape) == 2:
            self.data = self.data[xmin:xmax, ymin:ymax]
        else:
            self.data = self.data[xmin:xmax, ymin:ymax, :]

    def resize(self, height: int, width: int):
        if len(self.data.shape) == 2:
            # TODO: Rewrite that with skimage's resize function
            self.data = cv2.resize(self.data, (width, height), interpolation=cv2.INTER_NEAREST)
        else:
            # Didn't find a better rescale function yet.
            new_image_data = np.zeros((height, width, self.data.shape[2]), dtype=self.data.dtype)
            for i in range(self.data.shape[2]):
                # TODO: Rewrite that with skimage's resize function
                new_image_data[:, :, i] = cv2.resize(self.data[:, :, i], (width, height), interpolation=cv2.INTER_NEAREST)
            self.data = new_image_data

    def pad(self, padding: Tuple[int, int, int, int], constant_values: int = 0):
        left, right, top, bottom = padding
        if len(self.data.shape) == 2:
            self.data = np.pad(self.data, ((top, bottom), (left, right)), constant_values=constant_values)
        else:
            # Assume 3 dimensions
            self.data = np.pad(self.data, ((top, bottom), (left, right), (0, 0)), constant_values=constant_values)

    def flip(self, axis: int =0):
        self.data = np.flip(self.data, axis=axis)

    def apply_transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
        transformed_image = self.transform(registerer, transformation, **kwargs)
        return Annotation(data=transformed_image,
                          labels=self.labels,
                          name=self.name)

    def copy(self):
        return Annotation(data=self.data.copy(),
                          labels=self.labels,
                          name=self.name)

    def store(self, path: str):
        # Use path as a directory here.
        create_if_not_exists(path)
        fname = 'annotations.nii.gz'
        img_path = join(path, fname)
        sitk.WriteImage(sitk.GetImageFromArray(self.data), img_path)
        if self.labels is not None:
            labels_path = join(path, 'labels.txt')
            with open(labels_path, 'w') as f:
                f.write('\n'.join(self.labels))
        additional_attributes = {
            'name' : self.name,
            'id': self._id
        }
        with open(join(path, 'additional_attributes.json'), 'w') as f:
            json.dump(additional_attributes, f)

    def get_by_label(self, label):
        if self.labels is None:
            return None
        idx = self.labels.index(label)
        return self.data[:, :, idx]

    def get_resolution(self) -> Optional[float]:
        return self.meta_information.get('resolution', None)

    @staticmethod
    def get_type() -> str:
        return 'annotation'

    @classmethod
    def load(cls, path):
        annotation = sitk.GetArrayFromImage(sitk.ReadImage(join(path, 'annotations.nii.gz')))
        labels_path = join(path, 'labels.txt')
        if exists(labels_path):
            with open(labels_path) as f:
                labels = [x.strip() for x in f.readlines()]
        else:
            labels = None
        with open(join(path, 'additional_attributes.json')) as f:
            additional_attributes = json.load(f)
        name = additional_attributes['name']
        id_ = additional_attributes['id']
        annotation = cls(data=annotation, labels=labels, name=name)
        annotation._id = id_
        return annotation