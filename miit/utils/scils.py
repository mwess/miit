"""
    Utility functions for SCiLS formats
"""

import json
from typing import List, Tuple

import numpy as np
import pyimzml

from miit.spatial_data.base_types import Annotation

def read_srd_with_msi(path: str, 
                      msi: pyimzml.ImzMLParser.ImzMLParser, 
                      target_resolution: float = 1) -> Annotation:
    """Read srd file using spatial information encoded in 

    Args:
        path (str): Path to srd file.
        msi (pyimzml.ImzMLParser.ImzMLParser): Source msi file. Used as a basis for 
            scaling and target image size.
        target_resolution (float, optional): Target resolution. Does not adjust for the unit used. 
            Defaults to 1.

    Returns:
        Annotation: _description_
    """
    scale_x = msi.imzmldict['pixel size x']/target_resolution
    scale_y = msi.imzmldict['pixel size y']/target_resolution
    max_x = int(msi.imzmldict['max dimension x']/target_resolution)
    max_y = int(msi.imzmldict['max dimension y']/target_resolution)
    return read_srd(path, (max_x, max_y), (scale_x, scale_y))

def read_srd(path: str, 
             dims: Tuple[int, int],
             scale: Tuple[float, float] = None 
             ) -> List[Annotation]:
    """Read srd file and translate into image format.

    Args:
        path (str): Path to srd file.
        dims (Tuple[int, int]): Size of target image space.
        scale (Tuple[int, int]): Scaling applied to each datapoint in srd. Defaults to (1, 1).
    
    """
    if scale is None:
        scale = (1, 1)
    with open(path, 'rb') as f:
        srd = json.load(f)
    scale_x, scale_y = scale
    max_x, max_y = dims
    annotations = []
    for region in srd['Regions']:
        points = []
        for source in region['Sources']:
            for point in source['Spots']:
        # for _, point in enumerate(srd['Regions'][0]['Sources'][0]['Spots']):
                x = point['X']
                y = point['Y']
                points.append((x,y))
        points = np.array(points)
        # needs to be 0 translated to fit into MSI spatial layout
        points[:, 0] = points[:,0] - np.min(points[:,0])
        points[:, 1] = points[:,1] - np.min(points[:,1])
        annotation_mat = np.zeros((max_y, max_x), dtype=np.uint8)        
        for i in range(points.shape[0]):
            x = points[i,0]
            y = points[i,1]
            x_s = int(x*scale_x)
            x_e = int(x_s + scale_x)
            y_s = int(y*scale_y)
            y_e = int(y_s + scale_y)
            annotation_mat[y_s:y_e,x_s:x_e] = 1
        annotation = Annotation(data=annotation_mat, name=f"""{region['Name']}""")
        annotation.meta_information['source_file'] = path
        annotations.append(annotation)
    return annotations