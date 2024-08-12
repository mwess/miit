import cv2
import tifffile


from .base_imaging import BaseImage, BasePointset
from .annotation import Annotation
from .default_image import DefaultImage
from .geojson import GeoJSONData
from .pointset import Pointset
from .ometiff_image import OMETIFFImage


def read_image(fpath):
    if fpath.endswith('.tiff'):
        tiff_file = tifffile.TiffFile(fpath)
        return DefaultImage(data=tiff_file.asarray())
        # Read tiffil.
    else:
        img =  cv2.imread(fpath)
        image = DefaultImage(data=img)
        return image