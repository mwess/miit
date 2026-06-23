import copy
import gzip
import json
import os
import uuid
from dataclasses import dataclass, field
from os.path import join
from typing import Any, Callable
from zipfile import ZipFile

import geojson
from geojson import FeatureCollection
import numpy, numpy as np
import shapely

from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.spatial_data.base_classes.base_imaging import BasePointset
from miit.spatial_data.base_classes import MIITobject
from miit.utils.utils import create_if_not_exists


@MIITobject
@dataclass(kw_only=True)
class GeoJSONData(BasePointset):

    # TODO:  Rewrite geojson_data to data
    data: geojson.GeoJSON
    _id: uuid.UUID = field(init=False)
    name: str = ''
    feature_fields_to_transform: list[str] | str = 'geometry'

    def __post_init__(self) -> None:
        self._id = uuid.uuid1()

    @staticmethod
    def make_deep_feature_copy(feature: geojson.Feature, exclude_features: str | list[str] = 'geometry'):
        if isinstance(exclude_features, str):
            exclude_features = [exclude_features]
        new_feature = geojson.Feature()
        for key in feature.keys():
            if key in exclude_features:
                continue
            new_feature[key] = copy.deepcopy(feature[key])
        return new_feature

    @staticmethod
    def fix_geometry(geometry, 
                     throw_exception_for_invalid_geometry: bool = False):
        if not geometry.is_valid:
            geometry = geometry.buffer(0)
            if not geometry.is_valid and throw_exception_for_invalid_geometry:
                raise Exception('Geometry could not be fixed.')
        return geometry

    def _apply_transform(self,
                         function: Callable,
                         fix_warped_geometry: bool = True,
                         override_feature_fields_to_transform: list[str] | str | None = None
                         ) -> geojson.FeatureCollection | geojson.GeoJSON:
        features = self.data['features'] if 'features' in self.data else self.data
        feature_fields_to_transform = self.feature_fields_to_transform if override_feature_fields_to_transform is None else override_feature_fields_to_transform
        if isinstance(feature_fields_to_transform, str):
            feature_fields_to_transform = [feature_fields_to_transform]
        warped_features = []
        for idx, feature in enumerate(features):
            try:
                warped_feature = GeoJSONData.make_deep_feature_copy(feature, exclude_features=feature_fields_to_transform)
                for feature_field in feature_fields_to_transform:
                    geometry = feature[feature_field]
                    if geometry is not None:
                        warped_geometry = shapely.transform(shapely.from_geojson(str(geometry)), function)
                        if fix_warped_geometry:
                            warped_geometry = GeoJSONData.fix_geometry(warped_geometry)
                        warped_geometry = geojson.loads(shapely.to_geojson(warped_geometry))
                        warped_feature[feature_field] = warped_geometry
                    else:
                        warped_feature[feature_field] = None
                warped_features.append(warped_feature)
            except Exception as e:
                print(e)
                print(f'Index: {idx}')
                print(feature)
                raise e    
        if isinstance(self.data, geojson.FeatureCollection):
            warped_data = FeatureCollection(warped_features)
            for key in self.data:
                if key != 'features':
                    warped_data[key] = copy.deepcopy(self.data[key])
        else:
            warped_data = warped_features
        return warped_data

    def apply_transform(self, 
                        registerer: Registerer, 
                        transformation: RegistrationResult, 
                        fix_warped_geometry: bool = True, 
                        **kwargs: dict) -> Any:
        # geometries = self.data['features'] if 'features' in self.data else self.data
        # geometries = self.data if isinstance(self.data, list) else self.data['features']
        is_feature_collection = isinstance(self.data, geojson.FeatureCollection)
        features = self.data['features'] if is_feature_collection else self.data
        warped_features = []
        for idx, feature in enumerate(features):
            # copied_geometry = copy.deepcopy(geometry)
            try:
                warped_feature = GeoJSONData.make_deep_feature_copy(feature)
                if feature['geometry'] is not None:
                    # warped_geometry = shapely.transform(shapely.from_geojson(str(feature['geometry'])), lambda x: registerer.transform_pointset(np.array([x[:0], x[:1]]), transformation))
                    warped_geometry = shapely.transform(shapely.from_geojson(str(feature['geometry'])), lambda x: registerer.transform_pointset(x, transformation, **kwargs))
                    if fix_warped_geometry:
                        if not warped_geometry.is_valid:
                            warped_geometry = warped_geometry.buffer(0)
                            if not warped_geometry.is_valid:
                                # Print a warning here.
                                pass
                    warped_geometry = geojson.loads(shapely.to_geojson(warped_geometry))
                    warped_feature['geometry'] = warped_geometry
                else:
                    warped_feature['geometry'] = None
                # warped_geometry = geojson.utils.map_tuples(lambda coords: warp_geojson_coord_tuple__(coords, registerer, transformation), copied_geometry)
                warped_features.append(warped_feature)
            except Exception as e:
                print(e)
                print(f'Index: {idx}')
                print(feature)
                raise e
        if is_feature_collection:
            warped_data = FeatureCollection(warped_features)
            for key in self.data:
                if key != 'features':
                    warped_data[key] = copy.deepcopy(self.data[key])
        else:
            warped_data = warped_features
        warped_geojson = GeoJSONData(data=warped_data, 
                                     name=self.name, 
                                     resolution=self.resolution)
        return warped_geojson

    # def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
    #     # TODO: Should anything outside max be removed?
    #     features = self.data['features'] if 'features' in self.data else self.data
    #     features_new = []
    #     for feature in features:
    #         feature_new = geojson.utils.map_tuples(lambda coords: [coords[0] - ymin, coords[1] - xmin], feature)
    #         features_new.append(feature_new)
    #     if 'features' in self.data:
    #         self.data['features'] = features_new
    #     else:
    #         self.data = features_new

    def crop(self, 
             xmin: int, 
             xmax: int, 
             ymin: int, 
             ymax: int,
             override_feature_fields_to_transform: list[str] | str | None = None
             ):
        # TODO: Should anything outside max be removed?
        transform_function = lambda coords: coords - np.array([ymin, xmin])
        warped_data = self._apply_transform(transform_function, override_feature_fields_to_transform=override_feature_fields_to_transform)
        self.data = warped_data

    def resize(self, 
               width: int, 
               height: int,
               reference_shape: tuple[int, int],
               override_feature_fields_to_transform: list[str] | str | None = None):
        w_scale_factor = width/reference_shape[0]
        h_scale_factor = height/reference_shape[1]
        self.rescale((w_scale_factor, h_scale_factor), override_feature_fields_to_transform)


    # def resize(self, width: float, height: float):
    #     features = self.data['features'] if 'features' in self.data else self.data
    #     features_new = []
    #     for feature in features:
    #         # map_tuples seems to remove some keys such as isEllipse. 
    #         # Make a backup and overwrite every other feature except 
    #         # coordinates later.
    #         feature_new = copy.deepcopy(feature)
    #         feature_backup = geojson.utils.map_tuples(lambda coords: [coords[0] * width, coords[1] * height], feature)
    #         feature_new['geometry']['coordinates'] = feature_backup['geometry']['coordinates']
    #         features_new.append(feature_new)
    #     if 'features' in self.data:
    #         self.data['features'] = features_new
    #     else:
    #         self.data = features_new
    #     rate_w = 1 / width
    #     rate_h = 1 / height
    #     self.scale_resolution((rate_w, rate_h))        

    # def rescale(self, scaling_factor: float | tuple[float, float]):
    #     if isinstance(scaling_factor, float):
    #         scaling_factor = (scaling_factor, scaling_factor)
    #     w_scale_factor = scaling_factor[0]
    #     h_scale_factor = scaling_factor[1]
    #     features = self.data['features'] if 'features' in self.data else self.data
    #     features_new = []
    #     for feature in features:
    #         # map_tuples seems to remove some keys such as isEllipse. 
    #         # Make a backup and overwrite every other feature except 
    #         # coordinates later.
    #         feature_new = copy.deepcopy(feature)
    #         feature_backup = geojson.utils.map_tuples(lambda coords: [coords[0] * w_scale_factor, coords[1] * h_scale_factor], feature)
    #         feature_new['geometry']['coordinates'] = feature_backup['geometry']['coordinates']
    #         features_new.append(feature_new)
    #     if 'features' in self.data:
    #         self.data['features'] = features_new
    #     else:
    #         self.data = features_new
    #     rate_w = 1 / w_scale_factor
    #     rate_h = 1 / h_scale_factor
    #     self.scale_resolution((rate_w, rate_h))   

    def rescale(self, 
                scaling_factor: float | tuple[float, float],
                override_feature_fields_to_transform: list[str] | str | None = None):
        if isinstance(scaling_factor, float):
            scaling_factor = (scaling_factor, scaling_factor)
        w_scale_factor = scaling_factor[0]
        h_scale_factor = scaling_factor[1]
        transform_function = lambda coords: coords * np.array([w_scale_factor, h_scale_factor])
        warped_data = self._apply_transform(transform_function, override_feature_fields_to_transform=override_feature_fields_to_transform)
        self.data = warped_data
        rate_w = 1 / w_scale_factor
        rate_h = 1 / h_scale_factor
        self.scale_resolution((rate_w, rate_h))   

    # # TODO: Check error from resize as well
    # def pad(self, padding: tuple[int, int, int, int]):
    #     left, right, top, bottom = padding
    #     features = self.data['features'] if 'features' in self.data else self.data
    #     features_new = []
    #     for feature in features:
    #         feature_new = geojson.utils.map_tuples(lambda coords: [coords[0] + left, coords[1] + right], feature)
    #         features_new.append(feature_new)
    #     if 'features' in self.data:
    #         self.data['features'] = features_new
    #     else:
    #         self.data = features_new

    # TODO: Check error from resize as well
    def pad(self, 
            padding: tuple[int, int, int, int],
            override_feature_fields_to_transform: list[str] | str | None = None):
        left, right, _, _ = padding
        transform_function = lambda coords: coords + np.array([left, right])
        warped_data = self._apply_transform(transform_function, override_feature_fields_to_transform=override_feature_fields_to_transform)
        self.data = warped_data

    # def flip(self, ref_img_shape: tuple[int, int], axis: int = 0):
    #     features = self.data['features'] if 'features' in self.data else self.data
    #     features_new = []
    #     if axis == 1:
    #         center_x = ref_img_shape[1] // 2
    #         for feature in features:
    #             feature_new = geojson.utils.map_tuples(lambda coords: [coords[0] + 2 * (center_x - coords[0]), coords[1]], feature)
    #             features_new.append(feature_new)
    #     elif axis == 0:
    #         center_y = ref_img_shape[0] // 2
    #         for feature in features:
    #             feature_new = geojson.utils.map_tuples(lambda coords: [coords[0], coords[1] + 2 * (center_y - coords[1])], feature)
    #             features_new.append(feature_new)
    #     else:
    #         raise Exception(f"Cannot work with axis argument: {axis}")
    #     if 'features' in self.data:
    #         self.data['features'] = features_new
    #     else:
    #         self.data = features_new
    #     self.resolution = self.resolution[::-1]

    def flip(self, 
            ref_img_shape: tuple[int, int], 
            axis: int = 0,
            override_feature_fields_to_transform: list[str] | str | None = None):
        if axis == 1:
            center_x = ref_img_shape[1] // 2
            transform_function = lambda coords: coords + 2 * (np.stack(([np.repeat(center_x, coords.shape[0]), coords[:,1]])).squeeze() - coords)
        elif axis == 0:
            center_y = ref_img_shape[0] // 2
            transform_function = lambda coords: coords + 2 * (np.dstack((coords[:,0], np.repeat(center_y, coords.shape[0]))).squeeze() - coords)

        else:
            raise Exception(f"Invalid axis argument: {axis}")
        self.data = self._apply_transform(transform_function, override_feature_fields_to_transform=override_feature_fields_to_transform)
        self.resolution = (self.resolution[1], self.resolution[0])

    def copy(self):
        resolution = (self.resolution[0].copy(), self.resolution[1].copy())
        return GeoJSONData(data=self.data.copy(), 
                           name=self.name,
                           resolution=resolution)

    def store(self, path: str, use_id_as_subfolder: bool = True):
        create_if_not_exists(path)
        if use_id_as_subfolder:
            sub_path = join(path, str(self._id))
            create_if_not_exists(sub_path)
        else:
            sub_path = path
        fname = 'geojson_data.geojson'
        fpath = join(sub_path, fname)
        with open(fpath, 'w') as f:
            geojson.dump(self.data, f)
        attributes = {'name': self.name}
        with open(join(sub_path, 'attributes.json'), 'w') as f:
            json.dump(attributes, f)

    @staticmethod
    def get_type() -> str:
        return 'GeoJSONData'

    @classmethod
    def load(cls, path: str) -> 'GeoJSONData':
        id_ = uuid.UUID(os.path.basename(path.rstrip('/')))
        with open(join(path, str(id_), 'geojson_data.geojson')) as f:
            data = geojson.load(f)
        # geojson_data = read_geojson(join(path, 'geojson_data.geojson'))
        with open(join(path, str(id_), 'attributes.json')) as f:
            attributes = json.load(f)
        gdata = cls(data=data, name=attributes.get('name', ''))
        gdata._id = id_
        return gdata

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
        if path_to_geojson.endswith('.geojson'):
            with open(path_to_geojson) as f:
                data = geojson.load(f)
        elif path_to_geojson.endswith('.gz'):
            with gzip.open(path_to_geojson) as f:
                data = geojson.load(f)
        elif path_to_geojson.endswith('.zip'):
            with ZipFile(path_to_geojson) as f:
                fname = f.namelist()[0]
                fcont = f.read(fname)
                data = geojson.loads(fcont)
        return cls(data=data, name=name)
    
def warp_geojson_coord_tuple__(coord: tuple[float, float] | tuple[numpy.float64, numpy.float64], 
                               registerer: Registerer, 
                               transform: RegistrationResult) -> tuple[float, float]:
    """Transforms coordinates from geojson data from moving to fixed image space.

    Args:
        coord (Tuple[float, float]): 
        transform (SimpleITK.SimpleITK.Transform): 

    Returns:
        Tuple[float, float]: 
    """
    ps = np.array([(coord[0], coord[1])]).astype(float)
    # ps = np.array([(float(coord[0]), float(coord[1]))]).astype(float)
    warped_ps = registerer.transform_pointset(ps, transform)
    return (warped_ps[0, 0], warped_ps[0, 1])
