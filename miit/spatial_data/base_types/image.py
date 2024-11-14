import json
import uuid
from dataclasses import dataclass
from os.path import join
from typing import Any, ClassVar


import cv2
import numpy as np
import SimpleITK as sitk


from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.spatial_data.base_types.base_imaging import BaseImage
from miit.utils.utils import create_if_not_exists
from miit.utils.distance_unit import DUnit


@dataclass(kw_only=True)
class Image(BaseImage):

    interpolation_mode: ClassVar[str] = 'LINEAR'

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.data = self.data[xmin:xmax, ymin:ymax]

    def resize(self, width: int, height: int):
        # Use opencv's resize function here, because it typically works a lot faster and for now
        # we assume that data in Image is always some kind of rgb like image.
        old_width, old_height = self.data.shape[:2]
        self.data = cv2.resize(self.data, (height, width))
        rate_w = old_width / width
        rate_h = old_height / height
        self.scale_resolution((rate_w, rate_h))

    def rescale(self, scaling_factor: float | tuple[float, float]):
        if isinstance(scaling_factor, float):
            scaling_factor = (scaling_factor, scaling_factor)
        w, h = self.data.shape[:2]
        w_n, h_n = int(w*scaling_factor[0]), int(h*scaling_factor[1])
        self.resize(w_n, h_n)

    def pad(self, padding: tuple[int, int, int, int], constant_values: int = 0):
        left, right, top, bottom = padding
        self.data = cv2.copyMakeBorder(self.data, top, bottom, left, right, cv2.BORDER_CONSTANT, constant_values)

    def flip(self, axis: int = 0):
        self.data = np.flip(self.data, axis=axis)
        self.resolution = self.resolution[::-1]

    def apply_transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: dict) -> Any:
        transformed_image = self.transform(registerer, transformation, **kwargs)
        return Image(data=transformed_image, resolution=self.resolution)
    
    def copy(self):
        return Image(data=self.data.copy(), resolution = self.resolution)

    def store(self, path: str):
        create_if_not_exists(path)
        fname = 'image.nii.gz'
        img_path = join(path, fname)
        attributes = {
            'name': self.name,
            'id': str(self._id),
            'resolution': {
                'width': self.resolution[0].to_json(),
                'height': self.resolution[1].to_json()
            }
        }
        with open(join(path, 'attributes.json'), 'w') as f:
            json.dump(attributes, f)
        sitk.WriteImage(sitk.GetImageFromArray(self.data), img_path)

    @staticmethod
    def get_type() -> str:
        return 'image'

    @classmethod
    def load(cls, path: str) -> 'Image':
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(path, 'image.nii.gz')))
        with open(join(path, 'attributes.json')) as f:
            attributes = json.load(f)
        name = attributes['name']
        id_ = uuid.UUID(attributes['id'])
        resolution = attributes['resolution']
        res_w = DUnit.from_dict(resolution['width'])
        res_h = DUnit.from_dict(resolution['height'])
        image = cls(data=img, name=name, resolution=(res_w, res_h))
        image._id = id_
        return image

    @classmethod
    def load_from_path(cls, path: str, name: str = '') -> 'Image':
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))
        return cls(data=img, name=name)
