import json
import os
import uuid
from dataclasses import dataclass
from os.path import join
from typing import Any, ClassVar, Dict, Optional, Tuple


import cv2
import numpy, numpy as np
import SimpleITK as sitk


from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.spatial_data.base_types.base_imaging import BaseImage
from miit.utils.utils import create_if_not_exists


@dataclass(kw_only=True)
class DefaultImage(BaseImage):

    interpolation_mode: ClassVar[str] = 'LINEAR'

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.data = self.data[xmin:xmax, ymin:ymax]

    def resize(self, width, height):
        # Use opencv's resize function here, because it typically works a lot faster and for now
        # we assume that data in Image is always some kind of rgb like image.
        self.data = cv2.resize(self.data, (height, width))

    def pad(self, padding: Tuple[int, int, int, int], constant_values: int = 0):
        left, right, top, bottom = padding
        self.data = cv2.copyMakeBorder(self.data, top, bottom, left, right, cv2.BORDER_CONSTANT, constant_values)

    def flip(self, axis: int = 0):
        self.data = np.flip(self.data, axis=axis)

    def apply_transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
        transformed_image = self.transform(registerer, transformation, **kwargs)
        return DefaultImage(data=transformed_image)

    def copy(self):
        return DefaultImage(data=self.data.copy())

    def store(self, path: str):
        create_if_not_exists(path)
        fname = 'image.nii.gz'
        img_path = join(path, fname)
        attributes = {
            'name': self.name,
            'id': str(self._id)
        }
        with open(join(path, 'attributes.json'), 'w') as f:
            json.dump(attributes, f)
        sitk.WriteImage(sitk.GetImageFromArray(self.data), img_path)

    def get_resolution(self) -> Optional[float]:
        return self.meta_information.get('resolution', None)

    @staticmethod
    def get_type() -> str:
        return 'default_image'

    @classmethod
    def load(cls, path: str):
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(path, 'image.nii.gz')))
        with open(join(path, 'attributes.json')) as f:
            attributes = json.load(f)
        name = attributes['name']
        id_ = uuid.UUID(attributes['id'])
        image = cls(data=img, name=name)
        image._id = id_
        return image