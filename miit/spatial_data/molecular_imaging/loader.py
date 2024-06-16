from dataclasses import dataclass, field
from typing import Dict

from miit.spatial_data.molecular_imaging.scils_export_imzml import ScilsExportImzml
from miit.spatial_data.molecular_imaging.visium10x import Visium10X
from miit.spatial_data.molecular_imaging.imaging_data import BaseMolecularImaging


class MolecularImagingLoaderException(Exception):
    
    pass


@dataclass
class MolecularImagingLoader:
    
    class_map: Dict = field(init=False)
    
    def __post_init__(self):
        self.class_map = {}
        self.class_map[ScilsExportImzml.get_type()] = ScilsExportImzml
        self.class_map[Visium10X.get_type()] = Visium10X

    def load(self, data_type, path):
        if data_type not in self.class_map:
            raise MolecularImagingLoaderException(f'data_type {data_type} not found in loader.')
        return self.class_map[data_type].load(path)

    def add_class(self, clazz: BaseMolecularImaging):
        self.class_map[clazz.get_type()] = clazz
        
    @staticmethod
    def load_default_loader():
        return MolecularImagingLoader()
        
    
    