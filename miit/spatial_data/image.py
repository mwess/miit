import abc
from dataclasses import dataclass, field
import json
from typing import Any, ClassVar, Optional, Protocol, Tuple, List, Dict
import os
from os.path import join, exists
import uuid
import xml.etree.ElementTree as ET

import cv2
import geojson
import numpy
import numpy as np
import pandas
import pandas as pd
import SimpleITK as sitk
import tifffile
import shapely
import skimage

from greedyfhist.utils.geojson_utils import geojson_2_table, convert_table_2_geo_json, read_geojson
from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.utils import create_if_not_exists

def convert_linestring_to_polygon(geom):
    coords = np.array(list(geom.coords))
    if not geom.is_closed:
        coords = np.vstack((coords, coords[0, :]))
    return shapely.Polygon(coords)

@dataclass
class SupportsImageInterpolation(Protocol):

    image_data: numpy.array
    interpolation_mode: str

    def transform_image(self, registerer: Registerer, transformation: RegistrationResult, args: Optional[Any] = None) -> Any:
        ...

def read_image(fpath):
    if fpath.endswith('.tiff'):
        tiff_file = tifffile.TiffFile(fpath)
        # resolution = get_resolution_from_tiff(tiff_file)
        return Image(data=tiff_file.asarray())
        # Read tiffil.
    else:
        img =  cv2.imread(fpath)
        image = Image(data=img)
        return image
    

def get_resolution_from_tiff(tiff_file):
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    root = ET.fromstring(tiff_file.ome_metadata)  
    node = root.findall('ome:Image', ns)[0].findall('ome:Pixels', ns)[0]
    x = float(node.attrib['PhysicalSizeX'])
    y = float(node.attrib['PhysicalSizeY'])
    return (x + y)/2

@dataclass(kw_only=True)
class BaseImage(abc.ABC):

    data: numpy.array
    interpolation_mode: ClassVar[str]
    id_:uuid.UUID = field(init=False)

    def __post_init__(self) -> None:
        self.id_ = uuid.uuid1()

    def transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
        image_transformed = registerer.transform_image(self.data, transformation, self.interpolation_mode, **kwargs)
        return image_transformed

    @abc.abstractmethod
    def apply_bounding_parameters(self, xmin: int, xmax: int, ymin: int, ymax: int):
        pass

    @abc.abstractmethod
    def rescale(self, height: int, width: int):
        pass

    @abc.abstractmethod
    def pad(self, padding: Tuple[int, int, int, int], constant_values: int = 0):
        pass

    @abc.abstractmethod
    def flip(self, axis: int =0):
        pass

    @abc.abstractmethod
    def warp(self, registerer: Registerer, transformation: Any, **kwargs: Dict) -> Any:
        pass

    @abc.abstractmethod
    def store(self, path: str):
        pass

    @classmethod
    @abc.abstractmethod
    def load(path: str):
        pass


@dataclass(kw_only=True)
class Image(BaseImage):
    
    interpolation_mode: ClassVar[str] = 'LINEAR'

    def apply_bounding_parameters(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.data = self.data[xmin:xmax, ymin:ymax]

    def rescale(self, height, width):
        # TODO: Replace that with skimage's resize function.
        self.data = cv2.resize(self.data, (width, height))

    def pad(self, padding: Tuple[int, int, int, int], constant_values: int = 0):
        left, right, top, bottom = padding
        # TODO: Find out whether we can replace that with a function from skimage.
        self.data = cv2.copyMakeBorder(self.data, top, bottom, left, right, cv2.BORDER_CONSTANT, constant_values)

    def flip(self, axis: int =0):
        self.data = np.flip(self.data, axis=axis)

    def warp(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
        transformed_image = self.transform(registerer, transformation, **kwargs)
        return Image(data=transformed_image)

    def copy(self):
        return Image(data=self.data.copy())

    def store(self, path: str):
        sub_path = join(path, str(self.id_))
        create_if_not_exists(sub_path)
        fname = 'image.nii.gz'
        img_path = join(sub_path, fname)
        sitk.WriteImage(sitk.GetImageFromArray(self.data), img_path)

    @classmethod
    def load(cls, path: str):
        img = sitk.GetArrayFromImage(sitk.ReadImage(join(path, 'image.nii.gz')))
        image = cls(data=img)
        id_ = uuid.UUID(os.path.basename(path.rstrip('/')))
        image.id_ = id_
        return image
        

# TODO: Fix orientation from C x H x W to H x W x C.
@dataclass(kw_only=True)
class Annotation(BaseImage):
    """
    Assumes annotations are stored as H x W x C.
    """

    interpolation_mode: ClassVar[str] = 'NN'
    labels: List[str] = None
    name: str = ''

    def apply_bounding_parameters(self, xmin: int, xmax: int, ymin: int, ymax: int):
        # TODO: Add check for image bounds
        if len(self.data.shape) == 2:
            self.data = self.data[xmin:xmax, ymin:ymax]
        else:
            self.data = self.data[xmin:xmax, ymin:ymax, :]

    def rescale(self, height: int, width: int):
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

    def warp(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
        transformed_image = self.transform(registerer, transformation, **kwargs)
        return Annotation(data=transformed_image, 
                          labels=self.labels,
                          name=self.name)

    def copy(self):
        """
        Warning! Returns a new id.
        """
        return Annotation(data=self.data.copy(),
                          labels=self.labels,
                          name=self.name)

    def store(self, path: str):
        # Use path as a directory here.
        create_if_not_exists(path)
        sub_path = join(path, str(self.id_))
        create_if_not_exists(sub_path)
        fname = 'annotations.nii.gz'
        img_path = join(sub_path, fname)
        sitk.WriteImage(sitk.GetImageFromArray(self.data), img_path)
        if self.labels is not None:
            labels_path = join(sub_path, 'labels.txt')
            with open(labels_path, 'w') as f:
                f.write('\n'.join(self.labels))
        additional_attributes = {
            'name' : self.name
        }
        with open(join(sub_path, 'additional_attributes.json'), 'w') as f:
            json.dump(additional_attributes, f)

    def get_by_label(self, label):
        if self.labels is None:
            return None
        idx = self.labels.index(label)
        return self.data[:, :, idx]

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
        annotation = cls(data=annotation, labels=labels, name=name)
        id_ = uuid.UUID(os.path.basename(path.rstrip('/')))
        annotation.id_ = id_
        return annotation


@dataclass(kw_only=True)
class Pointset:

    data: pandas.core.frame.DataFrame
    id_: uuid.UUID = field(init=False)
    name: str = ''

    def __post_init__(self) -> None:
        self.id_ = uuid.uuid1()

    def warp(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
        warped_pc = self.data.copy()
        pc_ = self.data[['x', 'y']].to_numpy()
        coordinates_transformed = registerer.transform_pointset(pc_, transformation, **kwargs)
        if isinstance(coordinates_transformed, np.ndarray):
            temp_df = pd.DataFrame(coordinates_transformed, columns=['x', 'y'])
            coordinates_transformed = temp_df
        # coordinates_transformed = transform_result.final_transform.pointcloud
        warped_pc = warped_pc.assign(x=coordinates_transformed['x'].values, y=coordinates_transformed['y'].values)
        return Pointset(data=warped_pc)

    def apply_bounding_parameters(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.data['x'] = self.data['x'] - ymin
        self.data['y'] = self.data['y'] - xmin

    def rescale(self, height_scale: float, width_scale: float):
        # Remember to convert new dimensions to scale.
        self.data['x'] = self.data['x'] * width_scale
        self.data['y'] = self.data['y'] * height_scale

    def pad(self, padding: Tuple[int, int, int, int]):
        left, right, top, bottom = padding
        self.data['x'] = self.data['x'] + left
        self.data['y'] = self.data['y'] + top

    def flip(self, ref_img_shape: Tuple[int, int], axis: int = 0):
        if axis == 0:
            center_x = ref_img_shape[1] // 2
            self.data.x = self.data.x + 2 * (center_x - self.data.x)
        elif axis == 1:
            center_y = ref_img_shape[0] // 2
            self.data.y = self.data.y + 2 * (center_y - self.data.y)
        else:
            pass
    
    def copy(self):
        return Pointset(data=self.data.copy())

    def store(self, path: str):
        create_if_not_exists(path)
        sub_path = join(path, str(self.id_))
        create_if_not_exists(sub_path)
        fname = 'pointset.csv'
        fpath = join(sub_path, fname)
        self.data.to_csv(fpath)

    @classmethod
    def load(cls, path: str):
        id_ = uuid.UUID(os.path.basename(path.rstrip('/')))
        df = pd.read_csv(join(path, 'pointset.csv'))
        ps = cls(data=df)
        ps.id_ = id_
        return ps
    
@dataclass(kw_only=True)
class GeoJson:

    # TODO:  Rewrite geojson_data to data
    geojson_data: geojson.GeoJSON
    data: pandas.core.frame.DataFrame = field(init=False)
    id_: uuid.UUID = field(init=False)
    name: str = ''

    def __post_init__(self) -> None:
        self.id_ = uuid.uuid1()
        self.data = geojson_2_table(self.geojson_data)

    def warp(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: Dict) -> Any:
        data_ann = Pointset(data=self.data)
        data_warped = data_ann.warp(registerer, transformation, **kwargs)
        warped_geojson = GeoJson(geojson_data=self.geojson_data, name=self.name)
        warped_geojson.data = data_warped.data
        return warped_geojson
        # warped_pc = self.data.copy()
        # pc_ = self.data[['x', 'y']]
        # coordinates_transformed = registerer.transform_pointset(pc_, transformation, **kwargs)
        # # coordinates_transformed = transform_result.final_transform.pointcloud
        # warped_pc = warped_pc.assign(x=coordinates_transformed['x'].values, y=coordinates_transformed['y'].values)
        
        # return GeoJson(warped_pc)

    def apply_bounding_parameters(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.data['x'] = self.data['x'] - ymin
        self.data['y'] = self.data['y'] - xmin

    def rescale(self, height_scale: float, width_scale: float):
        # Remember to convert new dimensions to scale.
        self.data['x'] = self.data['x'] * width_scale
        self.data['y'] = self.data['y'] * height_scale

    def pad(self, padding: Tuple[int, int, int, int]):
        left, right, top, bottom = padding
        self.data['x'] = self.data['x'] + left
        self.data['y'] = self.data['y'] + top

    def flip(self, ref_img_shape: Tuple[int, int], axis: int = 0):
        if axis == 0:
            center_x = ref_img_shape[1] // 2
            self.data.x = self.data.x + 2 * (center_x - self.data.x)
        elif axis == 1:
            center_y = ref_img_shape[0] // 2
            self.data.y = self.data.y + 2 * (center_y - self.data.y)
        else:
            pass
    
    def copy(self):
        geojson_data = convert_table_2_geo_json(self.data, self.geojson_data)
        return GeoJson(geojson_data=geojson_data)

    def store(self, path: str):
        create_if_not_exists(path)
        sub_path = join(path, str(self.id_))
        create_if_not_exists(sub_path)
        fname = 'geojson_data.geojson'
        fpath = join(sub_path, fname)
        geojson_data = convert_table_2_geo_json(self.data, self.geojson_data)
        with open(fpath, 'w') as f:
            geojson.dump(geojson_data)

    @classmethod
    def load(cls, path):
        id_ = uuid.UUID(os.path.basename(path.rstrip('/')))
        geojson_data = read_geojson(join(path, 'geojson_data.geojson'))
        gdata = cls(geojson_data=geojson_data)
        gdata.id_ = id_
        return gdata

    def get_geojson_data(self):
        return convert_table_2_geo_json(self.geojson_data, self.data)

    def convert_geojson_to_mask(self, ref_image: numpy.array):
        labels = []
        masks = []
        geojson_data = self.get_geojson_data()
        for feature in geojson_data['features']:
            # If no name can be found, ignore for now.
            if 'classification' not in feature['properties'] and 'name' not in feature['properties']:
                continue
            # Do not support cellcounts at the moment.
            # TODO: Add support for cells.
            if feature['properties']['objectType'] != 'annotation':
                continue
            geom = shapely.from_geojson(str(feature))
            # mask = np.zeros(ref_image.shape[:2], dtype=np.uint8)
            if geom.is_empty:
                print(f"""Feature of type {feature['geometry']['type']} is empty.""")
                continue
            if geom.geom_type == 'LineString':
                geom = convert_linestring_to_polygon(geom)
            if geom.geom_type == 'MultiPolygon':
                shape = (ref_image.shape[1], ref_image.shape[0])
                mask = np.zeros(shape, dtype=np.uint8)
                for geom_ in geom.geoms:
                    ext_coords = np.array(list(geom_.exterior.coords))
                    mask_ = skimage.draw.polygon2mask(shape, ext_coords)
                    mask[mask_] = 1
                    # poly_ = skimage.draw.polygon(ext_coords[:, 1], ext_coords[:, 0], ref_image.shape)
                    # mask[poly_[0], poly_[1]] = 1
            else:
                ext_coords = np.array(list(geom.exterior.coords))
                shape = (ref_image.shape[1], ref_image.shape[0])
                mask = skimage.draw.polygon2mask(shape, ext_coords)
                # poly_ = skimage.draw.polygon(ext_coords[:, 0], ext_coords[:, 1], ref_image.shape)
                # mask[poly_[0], poly_[1]] = 1
            mask = mask.transpose().astype(np.uint8)
            masks.append(mask)
            if 'classification' in feature['properties']:
                labels.append(feature['properties']['classification']['name'])
            else:
                labels.append(feature['properties']['name'])
            # TODO: Find a more generic way for this.
            # labels.append(feature['properties']['name'])
        annotation_mask = np.dstack(masks)
        # annotation_mask = np.moveaxis(annotation_mask, 2, 0)
        annotation = Annotation(data=annotation_mask, labels=labels, name=self.name)
        return annotation