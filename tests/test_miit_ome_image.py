import tempfile

import numpy as np
from bioio import BioImage
from bioio_base.types import PhysicalPixelSizes


from miit.spatial_data.base_types import OMEImage

from .utils import make_circle_image_multichannel

# def test_reading_ome_file():
#     page1_image = make_circle_image_multichannel(draw_c2=False)
#     page2_image = make_circle_image_multichannel(draw_c2=False)
#     page2_image[page2_image > 0] = 1
#     shape = (page1_image.shape[0], page1_image.shape[1], 3)
#     page3_image = np.ones(shape)
#     page_list = [page1_image, page2_image, page3_image]
#     # physical_pixel_sizes = PhysicalPixelSizes(Z=0,
#     #                                     X=1,
#     #                                     Y=1)
#     # physical_pixel_sizes = [physical_pixel_sizes] * len(page_list)
#     # channel_names = [['image1'], ['image2'], 'image3']
#     # dims_order = 'TCZYXS'
#     bio_image = BioImage(page_list[0])
#     # bio_image = BioImage(page_list, 
#     #                      channel_names=channel_names,
#     #                      dims_order=dims_order,
#     #                      physical_pixel_sizes=physical_pixel_sizes)



#     with tempfile.NamedTemporaryFile(suffix='.ome.tif') as f:
#         path = f.name
#         bio_image.save(path)