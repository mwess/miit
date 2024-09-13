import json
import os
import uuid
from dataclasses import dataclass, field
from os.path import join
from typing import Any, Dict, Optional, Tuple


import geojson
import numpy, numpy as np
import shapely
import skimage


from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.spatial_data.base_types.annotation import Annotation
from miit.spatial_data.base_types.base_imaging import BasePointset
from miit.utils.utils import create_if_not_exists


def convert_linestring_to_polygon(geom):
    coords = np.array(list(geom.coords))
    if not geom.is_closed:
        coords = np.vstack((coords, coords[0, :]))
    return shapely.Polygon(coords)


@dataclass(kw_only=True)
class GeoJSONData(BasePointset):

    # TODO:  Rewrite geojson_data to data
    data: geojson.GeoJSON
    _id: uuid.UUID = field(init=False)
    name: str = ''

    def __post_init__(self) -> None:
        self._id = uuid.uuid1()

    def apply_transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
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

    def pad(self, padding: Tuple[int, int, int, int]):
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

    def flip(self, ref_img_shape: Tuple[int, int], axis: int = 0):
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
            pass
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
    def load(cls, path):
        id_ = uuid.UUID(os.path.basename(path.rstrip('/')))
        with open(join(path, 'geojson_data.geojson')) as f:
            data = geojson.load(f)
        # geojson_data = read_geojson(join(path, 'geojson_data.geojson'))
        with open(join(path, 'attributes.json')) as f:
            attributes = json.load(f)
        gdata = cls(geojson_data=data, name=attributes.get('name', ''))
        gdata._id = id_
        return gdata

    def to_annotation(self,
                      ref_image: numpy.array,
                      label_fun: Optional[callable] = None) -> 'Annotation':
        """Utility function for converting a geojson object to an 
        annotation. 

        Args:
            ref_image (numpy.array):
                Defines the shape of the result annotation.

            label_fun (callable):
                Optional function for extracting label names from feature objects. 
                If not supplied, the labels of the annotation are taken from 'id',
                otherwise 'label_fun' is called.
        """
        labels = []
        masks = []
        geojson_data = self.data
        for feature in geojson_data['features']:
            # If no name can be found, ignore for now.
            # if 'classification' not in feature['properties'] and 'name' not in feature['properties']:
            #     continue
            # Do not support cellcounts at the moment.
            # TODO: Add support for cells.
            if feature['properties']['objectType'] != 'annotation':
                continue
            geom = shapely.from_geojson(str(feature))
            if geom.is_empty:
                print(f"""Feature of type {feature['geometry']['type']} is empty.""")
                continue
            if geom.geom_type == 'LineString':
                geom = convert_linestring_to_polygon(geom)
            if geom.geom_type == 'MultiPolygon':
                shape = (ref_image.shape[1], ref_image.shape[0])
                # print(f'Multipolygon: {shape}')
                mask = np.zeros(shape, dtype=np.uint8)
                for geom_ in geom.geoms:
                    ext_coords = np.array(list(geom_.exterior.coords))
                    mask_ = skimage.draw.polygon2mask(shape, ext_coords)
                    mask[mask_] = 1
                # print(f'Multipolygon mask: {mask.shape}')
            else:
                ext_coords = np.array(list(geom.exterior.coords))
                shape = (ref_image.shape[1], ref_image.shape[0])
                # print(f'Polygon: {shape}')
                mask = skimage.draw.polygon2mask(shape, ext_coords)
                # print(f'Polygon mask: {mask.shape}')
            mask = mask.transpose().astype(np.uint8)
            masks.append(mask)
            if label_fun is None:
                labels.append(feature['id'])
            else:
                labels.append(label_fun(feature))
        annotation_mask = np.dstack(masks)
        if len(annotation_mask.shape) > 2:
            is_multichannel = False
        else:
            is_multichannel = True
        annotation = Annotation(data=annotation_mask, labels=labels, name=self.name, is_multichannel=is_multichannel)
        return annotation

    def __warp_geojson_coord_tuple(self, coord: Tuple[float, float], registerer: Registerer, transform) -> Tuple[float, float]:
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