import cv2
import tifffile


from .base_imaging import BaseImage, BasePointset
from .annotation import Annotation
from .image import Image
from .geojson import GeoJSONData
from .pointset import Pointset
from .ome_image import OMEImage
from .ome_image import OMEImage
from .base_type_loader import SpatialBaseDataLoader


# Overwrite this function to include greedyfhists read_image functionality
def read_image(fpath):
    if fpath.endswith('.tiff'):
        tiff_file = tifffile.TiffFile(fpath)
        return Image(data=tiff_file.asarray())
        # Read tiffil.
    else:
        img =  cv2.imread(fpath)
        image = Image(data=img)
        return image