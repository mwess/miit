from miit.spatial_data.molecular_imaging.maldi import Maldi
from miit.spatial_data.molecular_imaging.spatial_transcriptomics import SpatialTranscriptomics
from miit.spatial_data.molecular_imaging.scils_export_imzml import ScilsExportImzml
from miit.spatial_data.molecular_imaging.visium10x import Visium10X

# TODO: Can that whole class be put in __init__.py ??
# TODO: Also replace loader function with loader class to make it easier to add other
# imaging types.


def load_molecular_imaging_data(config):

    if config['data_type'] == 'SpatialTranscriptomics':
        return SpatialTranscriptomics.from_config(config)
    elif config['data_type'] == 'Maldi':
        return Maldi.from_config(config)
    elif config['data_type'] == 'ScilsExportImzml':
        return ScilsExportImzml.from_config(config)
    elif config['data_type'] == 'Visium10X':
        return Visium10X.from_config(config)
    else:
        print(f"""data_type: {config['data_type']} not found.""")
    return None

    
# def load_molecular_imaging_data_from_directory(data_type, directory):
#     if data_type == 'SpatialTranscriptomics':
#         return SpatialTranscriptomics.from_directory(directory)
#     elif data_type == 'Maldi':
#         return Maldi.from_directory(directory)
#     elif data_type == 'ScilsExportImzml':
#         return ScilsExportImzml.from_directory(directory)
#     else:
#         print(f"""data_type: {data_type} not found.""")
#     return None

def load_molecular_imaging_data_from_directory(data_type, directory):
    if data_type == 'Visium10X':
        return Visium10X.load(directory)
    elif data_type == 'ScilsExportImzml':
        return ScilsExportImzml.load(directory)
    else:
        print(f"""data_type: {data_type} not found.""")
    return None