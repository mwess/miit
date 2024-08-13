from typing import Dict, Optional
import xml.etree.ElementTree as ET


import numpy
import tifffile


def get_tif_metadata(tif: tifffile.tifffile.TiffFile) -> Dict:
    root = ET.fromstring(tif.ome_metadata)
    ns = {'ns0': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    elem = root.findall('*/ns0:Pixels', ns)[0]
    metadata = elem.attrib
    # metadata['axes'] = 'TCYXS'
    del metadata['SizeX']
    del metadata['SizeY']
    del metadata['SizeC']
    del metadata['SizeT']
    del metadata['SizeZ']
    del metadata['DimensionOrder']
    return metadata       


def write_tif_file(path: str, data: numpy.array) -> None:
    tifffile.imwrite(path, data)


def write_ome_tif_file(path: str,
                       data: numpy.array,
                       metadata: Dict,
                       options: Optional[Dict] = None) -> None:
    if options is None:
        if len(data.shape) == 2:
            options = {}
        else:
            options = dict(photometric='rgb')
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        tif.write(
            data,
            metadata=metadata,
            **options
        )    