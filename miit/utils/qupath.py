from typing import Callable

import cv2
import geojson
import numpy, numpy as np
import shapely

from miit.spatial_data.base_types import Annotation, GeoJSONData, BaseImage


def convert_linestring_to_polygon(geom: shapely.LineString) -> shapely.Polygon:
    """Convert linestring to polygon.

    Args:
        geom (shapely.LineString): 

    Returns:
        shapely.Polygon:
    """
    coords = np.array(list(geom.coords))
    if not geom.is_closed:
        coords = np.vstack((coords, coords[0, :]))
    return shapely.Polygon(coords)


def fill_geojson_polygon(polygon: shapely.Polygon, img: numpy.ndarray, color: int = 1) -> numpy.ndarray:
    """Project filled polygon onto image template.

    Args:
        polygon (shapely.Polygon): 
        img (numpy.ndarray): 
        color (int, optional): Defaults to 1.

    Returns:
        numpy.ndarray: 
    """
    ext_coords = np.array(list(polygon.exterior.coords)).astype(int)
    cv2.fillPoly(img, pts=[ext_coords], color=color)
    for i in range(len(polygon.interiors)):
        int_coords = np.array(list(polygon.interiors[i].coords)).astype(int)
        cv2.fillPoly(img, pts=[int_coords], color=0)
    return img


def fill_geojson_multipolygon(multipolygon: shapely.MultiPolygon, img: numpy.ndarray, color: int = 1) -> numpy.ndarray:
    """Project filled multipolygon onto image template.

    Args:
        multipolygon (shapely.MultiPolygon): 
        img (numpy.ndarray): 
        color (int, optional): Defaults to 1.

    Returns:
        numpy.ndarray: 
    """
    for polygon in multipolygon.geoms:
        img = fill_geojson_polygon(polygon, img, color)
    return img


def resolve_naming_conflicts_fun(label: str, 
                                  existing_labels: list[str], 
                                  max_loop_size: int = 100000) -> str:
    """Helper function for retrieving unique label name.

    Args:
        label (str): 
        existing_labels (list[str]): 
        max_loop_size (int, optional): Defaults to 100000.

    Returns:
        str: 
    """
    new_label = label
    counter = 1
    while new_label in existing_labels and counter < max_loop_size:
        new_label = f'{label}_{counter}'
        counter += 1
    return new_label


def simple_naming_function(feature: geojson.Feature) -> str:
    """Derive name from geojson feature.

    Args:
        feature (geojson.Feature):

    Returns:
        str: 
    """
    try:
        return feature['properties']['classification']['name']
    except:
        return 'feature'


def geojson_to_annotation(geojson_data: GeoJSONData | geojson.FeatureCollection | geojson.Feature | list[geojson.Feature],
                  ref_image: BaseImage | numpy.ndarray,
                  label_fun: Callable[[geojson.Feature], str] | None = None,
                  name: str = '',
                  skip_non_polygons: bool = False,
                  resolve_name_conflicts: bool = True,
                  geometry: str = 'geometry',
                  select_object_type: list[str] | None = None,
                  keep_properties: bool = True,
                  ignore_invalid_geom_types: bool = True,
                  to_multichannel_annotation: bool = False) -> Annotation:
    """Utility function for converting a geojson object to an 
    annotation. 

    Args:
        geojson_data (GeoJSONData | geojson.feature.FeatureCollection):
            Data source.
        ref_image (Union[Image, numpy.ndarray]):
            Defines the shape of the result annotation.
        label_fun (Callable[[geojson.feature.Feature], str], optional):
            Optional function for extracting label names from feature objects. 
            If not supplied, the labels of the annotation are taken from 'id',
            otherwise 'label_fun' is called.
        name (str): Name given to returning annotation. Defaults to ''.
        skip_non_polygons (bool):
            If False, will try to convert any geom type that is not a polygon, to 
            a polygon. Defaults to False.
        geometry (str):
            Which geometry to convert. Can either be 'geometry' or 'nucleusGeometry'.
            Defaults to 'geometry'.
        select_object_type (list[str], optional):
            Chooses which object types to convert based on the object type in feature['properties']['objectType'].
            If empty or None, converts all object types to annotations. Otherwise, only converts object types 
            specified in 'select_object_type'.
        keep_properties (bool):
            If True, will return all properties in feature['properties'] in the Annotation
            attribute 'meta_information'.
        ignore_invalid_geom_types (bool):
            If True, geometry types that cannot be converted will simply be ignore. Otherwise,
            an exception is thrown.
        to_multichannel_annotation (bool):
            If True, will compress all converted annotations into a multichannel annotation. This can save memory
            and improve runtime, esp. for large images, but only one annotation per pixel is possible. Which
            annotation is chosen when multiple annotations occupy the same pixel is undefined. Defaults to False.
    Returns:
        Annotation:
    """
    if isinstance(ref_image, BaseImage):
        ref_image = ref_image.data
    if to_multichannel_annotation:
        labels = {}
        masks = np.zeros(ref_image.shape[:2])
    else:
        labels = []
        masks = []
    property_collection = {}
    if isinstance(geojson_data, GeoJSONData):
        geojson_data = geojson_data.data
    elif isinstance(geojson_data, geojson.Feature):
        geojson_data = {'features': [geojson_data]}
    elif isinstance(geojson_data, list):
        geojson_data = {'features': geojson_data}
    for idx, feature in enumerate(geojson_data['features']):
        if select_object_type and feature['properties']['objectType'] not in select_object_type:
            continue
        geom = shapely.from_geojson(str(feature[geometry]))
        if geom.is_empty:
            print(f"""Feature of type {feature[geometry]['type']} is empty.""")
            continue
        if geom.geom_type == 'LineString' and not skip_non_polygons:
            geom = convert_linestring_to_polygon(geom)
        if geom.geom_type == 'MultiPolygon':
            mask = np.zeros(ref_image.shape[:2], dtype=np.uint8)
            mask = fill_geojson_multipolygon(geom, mask, color=idx+1)
        elif geom.geom_type == 'Polygon':
            mask = np.zeros(ref_image.shape[:2])
            mask = fill_geojson_polygon(geom, mask, color=idx+1)
        else:
            if not ignore_invalid_geom_types:
                raise Exception(f'Geom type cannot be converted to annotation: {geom.geom_type}')
        if label_fun is None:
            label = feature['id']
        else:
            label = label_fun(feature)
        if resolve_name_conflicts:
            if to_multichannel_annotation:
                existing_labels = list(labels.keys())
            else:
                existing_labels = labels
            label = resolve_naming_conflicts_fun(label, existing_labels)
        if to_multichannel_annotation:
            inds = mask == idx + 1
            masks[inds] = mask[inds]
            labels[label] = idx + 1
        else:
            mask[mask > 0] = 1
            masks.append(mask)
            labels.append(label)
        if keep_properties:
            property_collection[label] = feature['properties']
    if to_multichannel_annotation:
        annotation_mask = masks
    else:
        annotation_mask = np.dstack(masks)
    meta_information = {'properties': property_collection} if keep_properties else {}
    annotation = Annotation(data=annotation_mask, 
                            labels=labels, 
                            name=name, 
                            is_multichannel=to_multichannel_annotation, 
                            meta_information=meta_information)
    return annotation