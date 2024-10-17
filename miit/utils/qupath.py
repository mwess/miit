from typing import Optional

import numpy, numpy as np
import shapely
import skimage

from miit.spatial_data.base_types import Annotation


def convert_linestring_to_polygon(geom):
    coords = np.array(list(geom.coords))
    if not geom.is_closed:
        coords = np.vstack((coords, coords[0, :]))
    return shapely.Polygon(coords)


# TODO: Split function up. Add function for extracting a single feature.
def to_annotation(geojson_data,
                  ref_image: numpy.ndarray,
                  label_fun: Optional[callable] = None,
                  name: str = '') -> Annotation:
    """Utility function for converting a geojson object to an 
    annotation. 

    Args:
        ref_image (numpy.array):
            Defines the shape of the result annotation.

        label_fun (callable):
            Optional function for extracting label names from feature objects. 
            If not supplied, the labels of the annotation are taken from 'id',
            otherwise 'label_fun' is called.
    
    Returns:
        Annotation:
    """
    labels = []
    masks = []
    for feature in geojson_data['features']:
        # If no name can be found, ignore for now.
        # if 'classification' not in feature['properties'] and 'name' not in feature['properties']:
        #     continue
        # Do not support cellcounts at the moment.
        # TODO: Add support for other fields.
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
            mask = np.zeros(shape, dtype=np.uint8)
            for geom_ in geom.geoms:
                ext_coords = np.array(list(geom_.exterior.coords))
                mask_ = skimage.draw.polygon2mask(shape, ext_coords)
                mask[mask_] = 1
        else:
            ext_coords = np.array(list(geom.exterior.coords))
            shape = (ref_image.shape[1], ref_image.shape[0])
            mask = skimage.draw.polygon2mask(shape, ext_coords)
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
    annotation = Annotation(data=annotation_mask, labels=labels, name=name, is_multichannel=is_multichannel)
    return annotation