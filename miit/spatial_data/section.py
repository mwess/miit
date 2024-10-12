from dataclasses import dataclass, field
import json
import os
from os.path import join, exists
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import shutil

import numpy, numpy as np

from miit.spatial_data.base_types import (
    Image,
    BaseImage,
    BasePointset,
    SpatialBaseDataLoader
)
from miit.spatial_data.spatial_omics import BaseSpatialOmics, SpatialOmicsDataLoader
from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.registerers.opencv_affine_registerer import OpenCVAffineRegisterer
from miit.spatial_data.base_types.geojson import GeoJSONData
from miit.utils.utils import copy_if_not_none, get_half_pad_size


# TODO: Refactor this out.
def get_boundary_box(image: numpy.array, background_value: float = 0) -> Tuple[int, int, int, int]:
    if len(image.shape) != 2:
        raise Exception(f'Bounding box requires a 2 dimensional array, but has: {len(image.shape)}')
    points = np.argwhere(image != background_value)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    return xmin, xmax, ymin, ymax        


def get_table_summary_string(section: 'Section') -> str:
    """
    Generates a summary string for all annotation and spatial omics data in a given section.

    Will look like this:

    ##############################################################################
    # Name             # ID                                   # Type             #
    ##############################################################################
    # msi neg ion mode # b9e7f19e-37e6-11ef-9af1-fa163eee643f # ScilsExportImzml #
    # landmarks        # 612d5b52-37e6-11ef-9af1-fa163eee643f # pointset         #
    # tissue_mask      # 612d01b6-37e6-11ef-9af1-fa163eee643f # annotation       #
    # msi neg ion mode # b9e7f19e-37e6-11ef-9af1-fa163eee643f # ScilsExportImzml #
    ##############################################################################
    """
    identifiers = []
    max_name_len = len('Name')
    max_id_len = len('ID')
    max_type_len = len('Type')
    for annotation in section.annotations + section.so_data:
        name_len = len(annotation.name)
        if max_name_len < name_len:
            max_name_len = name_len
        id_len = len(str(annotation._id))
        if max_id_len < id_len:
            max_id_len = id_len
        type_len = len(annotation.get_type())
        if max_type_len < type_len:
            max_type_len = type_len
        identifiers.append((annotation.name, str(annotation._id), annotation.get_type()))

    name_cell_len = max_name_len + 2
    id_cell_len = max_id_len + 2
    type_cell_len = max_type_len + 2

    table = []
    table_width = 1 + name_cell_len + 1 + id_cell_len + 1 + type_cell_len + 1
    table.append('#'*table_width)
    name = 'Name'
    id2 = 'ID'
    type_ = 'Type'

    l_name_pad, r_name_pad = get_half_pad_size(name, name_cell_len)
    l_id_pad, r_id_pad = get_half_pad_size(id2, id_cell_len)
    l_type_pad, r_type_pad = get_half_pad_size(type_, type_cell_len)
    header_line = '#' + ' ' * l_name_pad + name + ' ' * r_name_pad + '#' \
                + ' ' * l_id_pad + id2 + ' ' * r_id_pad + '#' \
                + ' ' * l_type_pad + type_ + ' ' * r_type_pad + '#'
    table.append(header_line)
    table.append('#'*table_width)
    for identifier in identifiers:
        name, id2, type_ = identifier
        l_name_pad, r_name_pad = get_half_pad_size(name, name_cell_len)
        l_id_pad, r_id_pad = get_half_pad_size(id2, id_cell_len)
        l_type_pad, r_type_pad = get_half_pad_size(type_, type_cell_len)
        line = '#' + ' ' * l_name_pad + name + ' ' * r_name_pad + '#' \
                    + ' ' * l_id_pad + id2 + ' ' * r_id_pad + '#' \
                    + ' ' * l_type_pad + type_ + ' ' * r_type_pad + '#'
        table.append(line)
    table.append('#'*table_width)
    return '\n'.join(table)


def groupwise_registration(sections: List['Section'],
                           registerer: Registerer,
                           **kwargs: Dict) -> Tuple[List['Section'], List[RegistrationResult]]:
    """
    Performs a groupwise registration on all supplied sections. A registration between all sections is computed
    and applied to all sections. Per convention, the last section in sections is used as the fixed section.
    
    Another convention: If custom masks are to be used, they need to be supplied in an annotation called tissue_mask.
    """
    # TODO: Allow to add options to the registerer.
    # TODO: Not all registerer can do groupwise registration. Find a way to filter those out.
    # Setup list with images
    images_with_mask_list = []
    for idx, section in enumerate(sections):
        image = section.reference_image.data
        # mask_annotation = section.segmentation_mask
        mask_annotation = section.get_annotations_by_names('tissue_mask')
        mask = mask_annotation.data if mask_annotation is not None else None
        images_with_mask_list.append((image, mask))
    # Now do groupwise registration.
    transforms, _ = registerer.groupwise_registration(images_with_mask_list, **kwargs)
    warped_sections = []
    for idx, section in enumerate(sections[:-1]):
        transform = transforms[idx]
        warped_section = section.apply_transform(registerer, transform)
        warped_sections.append(warped_section)
    warped_sections.append(sections[-1].copy())
    return warped_sections, transforms


def register_to_ref_image(target_image: numpy.array, 
                          source_image: numpy.array, 
                          data: Union[BaseImage, BasePointset],
                          registerer: Registerer = None,
                          **args) -> Tuple[Union[BaseImage, BasePointset], numpy.array]:
    """
    Finds a registration from source_image (or reference image) to target_image using registerer. 
    Registration is then applied to data. If registerer is None, will use the OpenCVAffineRegisterer as a default.
    args is additional options that will be passed to the registerer during registration.
    """
    if registerer is None:
        registerer = OpenCVAffineRegisterer()
    transformation = registerer.register_images(target_image, source_image, **args)
    warped_data = data.apply_transform(registerer, transformation)
    warped_ref_image = Image(data=source_image).apply_transform(registerer, transformation).data
    return warped_data, warped_ref_image


@dataclass
class Section:
    """
    Composite object that contains all spatial data belonging to one 
    sample section. It is assumed that all spatial data is aligned
    with the reference image. The reference image is used to register
    multiple sections with one another.
    Contains some utility functions for simple image transformations.
    Otherwise, consider using registerer/registration_results to apply
    complex image transformations to all data.
    
    
    Attributes
    ----------
    
    reference_image: BaseImage
        Used for registration with other sections. In some cases 
        used for spatial transformations on coordinate data.
        
    name: str
        Name of the section.
        
    id_: uuid.UUID = field(init=False)
        Internal id. Used for serialization.
                
    so_data: List[BaseMolecularImaging] = field(default_factory=lambda: [])
        Contains additional more complex spatial omics types (e.g. MSI, ST).
        
    annotations: List[Union[BaseImage, BasePointset]] = field(default_factory=lambda: [])
        List of additional annotations. 
    
    meta_information: Optional[Dict[Any, Any]] = None
        Additional meta information.
    """
    reference_image: Optional[BaseImage] = None
    name: Optional[str] = None
    _id: uuid.UUID = field(init=False)
    so_data: List[BaseSpatialOmics] = field(default_factory=lambda: [])
    annotations: List[Union[BaseImage, BasePointset]] = field(default_factory=lambda: [])
    meta_information: Optional[Dict[Any, Any]] = None


    def __post_init__(self):
        self._id = uuid.uuid1()

    def __hash__(self) -> int:
        return self._id

    def copy(self) -> 'Section':
        image = self.reference_image.copy()
        config = copy_if_not_none(self.meta_information)
        annotations = self.annotations.copy()
        so_data = self.so_data.copy()
        copied_section = Section(
            reference_image=image,
            name=self.name,
            annotations=annotations,
            so_data=so_data,
            meta_information=config)
        copied_section._id = self._id
        return copied_section

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.reference_image.crop(xmin, xmax, ymin, ymax)
        for annotation in self.annotations:
            annotation.crop(xmin, xmax, ymin, ymax)
        for so_data_ in self.so_data:
            so_data_.crop(xmin, xmax, ymin, ymax)

    def crop_by_mask(self, mask: numpy.ndarray):
        xmin, xmax, ymin, ymax = get_boundary_box(mask)
        self.crop(xmin, xmax, ymin, ymax)

    def pad(self, padding: Tuple[int, int, int, int]):
        self.reference_image.pad(padding)
        for annotation in self.annotations:
            annotation.pad(padding)
        for so_data_ in self.so_data:
            so_data_.pad(padding)

    def resize(self, width: int, height: int):
        w, h = self.reference_image.data[:2]
        ws = w // width
        hs = h // height
        self.reference_image.resize(width, height)
        for annotation in self.annotations:
            if isinstance(annotation, BasePointset):
                annotation.resize(ws, hs)
            else:
                annotation.resize(width, height)
        for so_data_ in self.so_data:
            so_data_.resize(width, height)

    def rescale(self, scaling_factor: float):
        pass

    def apply_transform(self, 
             registerer: Registerer, 
             transformation: RegistrationResult, 
             **kwargs: Dict) -> 'Section':
        """Applies transformation to all spatially resolved data in the section object.
        """
        image_transformed = self.reference_image.apply_transform(registerer, transformation, **kwargs)
        annotations_transformed = []
        for annotation in self.annotations:
            annotation_transformed = annotation.apply_transform(registerer, transformation, **kwargs)
            annotations_transformed.append(annotation_transformed)
        so_data_transformed_list = []
        for so_data_ in self.so_data:
            so_data_transformed = so_data_.apply_transform(registerer, transformation, **kwargs)
            so_data_transformed_list.append(so_data_transformed)
        
        config = self.meta_information.copy() if self.meta_information is not None else None
        transformed_section = Section(reference_image=image_transformed,
                       name=self.name,
                       annotations=annotations_transformed,
                       so_data=so_data_transformed_list,
                       meta_information=config)
        transformed_section._id = self._id
        return transformed_section

    def store(self, directory: str):
        # TODO: Rewrite that function.
        """
        Should look like this:
            File with mapping what attribute in what subfolder.
            Each subfolder is marked with their id.
        """
        if exists(directory):
            shutil.rmtree(directory)
        if not exists(directory):
            os.mkdir(directory)
        f_dict = {}
        f_dict['id'] = str(self._id)
        f_dict['name'] = self.name
        self.reference_image.store(join(directory, str(self.reference_image._id)))
        f_dict['reference_image'] = {
            'id': str(self.reference_image._id),
            'type': self.reference_image.get_type()
        }
        meta_information_path = join(directory, 'meta_information.json')
        with open(meta_information_path, 'w') as f:
            json.dump(self.meta_information, f)
            f_dict['meta_information_path'] = 'meta_information.json' 
        annotation_ids = []
        for annotation in self.annotations:
            annotation.store(join(directory, str(annotation._id)))
            annotation_ids.append(
                {
                    'id': str(annotation._id),
                    'type': annotation.get_type()
                }
            )
        f_dict['annotations'] = annotation_ids
        so_data_ids = []
        for so_data in self.so_data:
            so_dir = join(directory, str(so_data._id))
            so_data.store(so_dir)
            so_data_ids.append({
                'id': str(so_data._id),
                'type': so_data.get_type()
                }
            )
        f_dict['so_datas'] = so_data_ids
            
        with open(join(directory, 'attributes.json'), 'w') as f:
            json.dump(f_dict, f)

    def get_annotations_by_names(self, name: str):
        for annotation in self.annotations:
            if annotation.name == name:
                return annotation
        return None
    
    def get_annotation_by_id(self, id_: str):
        for annotation in self.annotations:
            if str(annotation._id) == id_:
                return annotation
        return None
    
    def print_additional_data_summary(self):
        print(get_table_summary_string(self))
            
    @classmethod
    def load(cls, directory: str, 
             base_type_loader: Optional[SpatialBaseDataLoader] = None,
             so_type_loader: Optional[SpatialOmicsDataLoader] = None) -> 'Section':
        """Loads a Section object from directory. 

        Args:
            directory (str): Source directory.
            base_type_loader (Optional[SpatialBaseDataLoader], optional): Data loader for base imaging types. Defaults to None.
            so_type_loader (Optional[SpatialOmicsDataLoader], optional): Data loader for spatial omics data types. Defaults to None.

        Returns:
            Section: _description_
        """
        if not exists(directory):
            # TODO: Throw custom error message
            pass
        if base_type_loader is None:
            base_type_loader = SpatialBaseDataLoader.load_default_loader()
        if so_type_loader is None:
            so_type_loader = SpatialOmicsDataLoader.load_default_loader()
        with open(join(directory, 'attributes.json')) as f:
            attributes = json.load(f)
        ref_image_dict = attributes['reference_image']
        ref_image_path = join(directory, ref_image_dict['id'])
        reference_image = base_type_loader.load(ref_image_dict['type'], ref_image_path)
        if 'meta_information_path' in attributes:
            meta_information_path = join(directory, attributes['meta_information_path'])
            with open(meta_information_path) as f:
                meta_information = json.load(f)
        else:
            meta_information = None
        annotations = []
        for annotation_dict in attributes['annotations']:
            sub_dir = annotation_dict['id']
            annotation = base_type_loader.load(annotation_dict['type'], join(directory, sub_dir))
            annotations.append(annotation)
        so_datas = []
        for so_dict in attributes['so_datas']:
            sub_dir = so_dict['id']
            so_data = so_type_loader.load(so_dict['type'],
                                  join(directory, sub_dir))
            so_datas.append(so_data)
        obj = cls(
            reference_image=reference_image,
            name=attributes['name'],
            annotations=annotations,
            so_data=so_datas,
            meta_information=meta_information
        )
        obj._id = uuid.UUID(attributes['id'])
        return obj

    def add_molecular_imaging_data(self,
                                   mi: BaseSpatialOmics,
                                   register_to_primary_image=True,
                                   reference_image: numpy.array = None,
                                   registerer: Optional[Registerer] = None):
        """
        Adds molecular imaging data to section.
        
        If register_to_primary_image is True, registers the molecular imaging data to the primary image.
        reference_image: Needs to be set it molecular imaging data is to be registered to the primary image.
        """
        # TODO: Add this function to loading from config!
        if register_to_primary_image:
            warped_mi = register_to_ref_image(self.reference_image.data, reference_image, mi, registerer)
        else:
            warped_mi = mi
        self.so_data.append(warped_mi)

    def flip(self, axis: int = 0):
        """
        Flip all spatial data in this section.
        """
        self.reference_image.flip(axis=axis)
        for annotation in self.annotations:
            if isinstance(annotation, BasePointset):
                annotation.flip(self.reference_image.data.shape[:2], axis=axis)
            else:
                annotation.flip(axis=axis)
        for so_data_ in self.so_data:
            so_data_.flip(axis=axis)