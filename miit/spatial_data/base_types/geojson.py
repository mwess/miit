import json
import os
import uuid
from dataclasses import dataclass, field
from os.path import join
from typing import Any


import geojson
import numpy, numpy as np
import shapely
import skimage


from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.spatial_data.base_types.annotation import Annotation
from miit.spatial_data.base_types.base_imaging import BasePointset
from miit.utils.utils import create_if_not_exists


@dataclass(kw_only=True)
class GeoJSONData(BasePointset):

    # TODO:  Rewrite geojson_data to data
    data: geojson.GeoJSON
    _id: uuid.UUID = field(init=False)
    name: str = ''

    def __post_init__(self) -> None:
        self._id = uuid.uuid1()

    def apply_transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: dict) -> Any:
        geometries = self.data['features'] if 'features' in self.data else self.data
        warped_geometries = []
        for _, geometry in enumerate(geometries):
            warped_geometry = geojson.utils.map_tuples(lambda coords: self.__warp_geojson_coord_tuple(coords, registerer, transformation), geometry)
            warped_geometries.append(warped_geometry)
        if 'features' in self.data:
            warped_data = self.data.copy()
            warped_data['features'] = warped_geometries
        else:
            warped_data = warped_geometries
        warped_geojson = GeoJSONData(data=warped_data, name=self.name)
        return warped_geojson

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        # TODO: Should anything outside max be removed?
        features = self.data['features'] if 'features' in self.data else self.data
        features_new = []
        for feature in features:
            feature_new = geojson.utils.map_tuples(lambda coords: [coords[0] - ymin, coords[1] - xmin], feature)
            features_new.append(feature_new)
        if 'features' in self.data:
            self.data['features'] = features_new
        else:
            self.data = features_new

    def resize(self, width: float, height: float):
        features = self.data['features'] if 'features' in self.data else self.data
        features_new = []
        for feature in features:
            feature_new = geojson.utils.map_tuples(lambda coords: [coords[0] * width, coords[1] * height], feature)
            features_new.append(feature_new)
        if 'features' in self.data:
            self.data['features'] = features_new
        else:
            self.data = features_new

    def rescale(self, scaling_factor: float):
        self.resize(scaling_factor, scaling_factor)

    def pad(self, padding: tuple[int, int, int, int]):
        left, right, top, bottom = padding
        features = self.data['features'] if 'features' in self.data else self.data
        features_new = []
        for feature in features:
            feature_new = geojson.utils.map_tuples(lambda coords: [coords[0] + left, coords[1] + right], feature)
            features_new.append(feature_new)
        if 'features' in self.data:
            self.data['features'] = features_new
        else:
            self.data = features_new

    def flip(self, ref_img_shape: tuple[int, int], axis: int = 0):
        features = self.data['features'] if 'features' in self.data else self.data
        features_new = []
        if axis == 0:
            center_x = ref_img_shape[1] // 2
            for feature in features:
                feature_new = geojson.utils.map_tuples(lambda coords: [coords[0] + 2 * (center_x - coords[0]), coords[1]], feature)
                features_new.append(feature_new)
        elif axis == 1:
            center_y = ref_img_shape[0] // 2
            for feature in features:
                feature_new = geojson.utils.map_tuples(lambda coords: [coords[0], coords[1] + 2 * (center_y - coords[1])], feature)
                features_new.append(feature_new)
        else:
            raise Exception(f"Cannot work with axis argument: {axis}")
        if 'features' in self.data:
            self.data['features'] = features_new
        else:
            self.data = features_new

    def copy(self):
        return GeoJSONData(data=self.data.copy(), name=self.name)

    def store(self, path: str):
        create_if_not_exists(path)
        sub_path = join(path, str(self._id))
        create_if_not_exists(sub_path)
        fname = 'geojson_data.geojson'
        fpath = join(sub_path, fname)
        with open(fpath, 'w') as f:
            geojson.dump(self.data)
        attributes = {'name': self.name}
        with open(join(sub_path, 'attributes.json'), 'w') as f:
            json.dump(attributes, f)

    @staticmethod
    def get_type() -> str:
        return 'geojson'

    @classmethod
    def load(cls, path: str) -> 'GeoJSONData':
        id_ = uuid.UUID(os.path.basename(path.rstrip('/')))
        with open(join(path, 'geojson_data.geojson')) as f:
            data = geojson.load(f)
        # geojson_data = read_geojson(join(path, 'geojson_data.geojson'))
        with open(join(path, 'attributes.json')) as f:
            attributes = json.load(f)
        gdata = cls(geojson_data=data, name=attributes.get('name', ''))
        gdata._id = id_
        return gdata

    def __warp_geojson_coord_tuple(self, 
                                   coord: tuple[float, float], 
                                   registerer: Registerer, 
                                   transform: RegistrationResult) -> tuple[float, float]:
        """Transforms coordinates from geojson data from moving to fixed image space.

        Args:
            coord (Tuple[float, float]): 
            transform (SimpleITK.SimpleITK.Transform): 

        Returns:
            Tuple[float, float]: 
        """
        ps = np.array([[coord[0], coord[1]]]).astype(float)
        warped_ps = registerer.transform_pointset(ps, transform)
        return (warped_ps[0, 0], warped_ps[0, 1])

    @classmethod
    def load_from_path(cls, 
                       path_to_geojson: str,
                       name: str = '') -> 'GeoJSONData':
        """Loads GeoJSONData objectom from path.

        Args:
            path_to_geojson (str): Path to geojson file.
            name (str, optional): Optional identifier. Defaults to ''.

        Returns:
            GeoJSONData: Initialized GeoJSONData object.
        """
        with open(path_to_geojson) as f:
            data = geojson.load(f)
        return cls(data=data, name=name)