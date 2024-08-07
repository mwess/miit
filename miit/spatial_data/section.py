from dataclasses import dataclass, field
import glob
import json
import os
from os.path import join, exists
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import shutil

import pandas as pd
import numpy
import numpy as np
import SimpleITK as sitk
from skimage import io

from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics
from miit.spatial_data.loaders import load_spatial_omics_data, SpatialDataLoader
from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.registerers.opencv_affine_registerer import OpenCVAffineRegisterer
from miit.spatial_data.image import BaseImage, DefaultImage, Annotation, Pointset, GeoJSONData
from miit.utils.utils import copy_if_not_none, get_half_pad_size


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
                           skip_deformable_registration: bool = False,
                           **kwarg: Dict):
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
    transforms, _ = registerer.groupwise_registration(images_with_mask_list, skip_deformable_registration=skip_deformable_registration)
    warped_sections = []
    for idx, section in enumerate(sections[:-1]):
        transform = transforms[idx]
        warped_section = section.apply_transform(registerer, transform)
        warped_sections.append(warped_section)
    warped_sections.append(sections[-1].copy())
    return warped_sections, transforms


def register_to_ref_image(target_image: numpy.array, 
                          source_image: numpy.array, 
                          data: Union[BaseImage, Pointset],
                          registerer=None,
                          args=None) -> Tuple[Union[BaseImage, Pointset], numpy.array]:
    """
    Finds a registration from source_image (or reference image) to target_image using registerer. 
    Registration is then applied to data. If registerer is None, will use the OpenCVAffineRegisterer as a default.
    args is additional options that will be passed to the registerer during registration.
    """
    if registerer is None:
        registerer = OpenCVAffineRegisterer()
    if args is None:
        args = {}
    transformation = registerer.register_images(target_image, source_image, **args)
    warped_data = data.apply_transform(registerer, transformation)
    warped_ref_image = DefaultImage(data=source_image).apply_transform(registerer, transformation).data
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
    
    reference_image: DefaultImage
        Used for registration with other sections. In some cases 
        used for spatial transformations on coordinate data.
        
    name: str
        Name of the section.
        
    id_: uuid.UUID = field(init=False)
        Internal id. Used for serialization.
                
    so_data: List[BaseMolecularImaging] = field(default_factory=lambda: [])
        Contains additional more complex spatial omics types (e.g. MSI, ST).
        
    annotations: List[Union[Annotation, Pointset, GeoJson]] = field(default_factory=lambda: [])
        List of additional annotations. 
    
    meta_information: Optional[Dict[Any, Any]] = None
        Additional meta information.
    """
    reference_image: Optional[DefaultImage] = None
    name: Optional[str] = None
    _id: uuid.UUID = field(init=False)
    so_data: List[BaseSpatialOmics] = field(default_factory=lambda: [])
    annotations: List[Union[DefaultImage, Annotation, Pointset, GeoJSONData]] = field(default_factory=lambda: [])
    meta_information: Optional[Dict[Any, Any]] = None


    def __post_init__(self):
        self._id = uuid.uuid1()

    def __hash__(self) -> int:
        return self._id

    def copy(self):
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

    def crop(self, xmin, xmax, ymin, ymax):
        self.reference_image.crop(xmin, xmax, ymin, ymax)
        for annotation in self.annotations:
            annotation.crop(xmin, xmax, ymin, ymax)
        for so_data_ in self.so_data:
            so_data_.crop(xmin, xmax, ymin, ymax)

    def crop_by_mask(self, mask):
        xmin, xmax, ymin, ymax = get_boundary_box(mask)
        self.crop(xmin, xmax, ymin, ymax)

    def pad(self, padding: Tuple[int, int, int, int]):
        self.reference_image.pad(padding)
        for annotation in self.annotations:
            annotation.pad(padding)
        for so_data_ in self.so_data:
            so_data_.pad(padding)

    def resize(self, height: int, width: int):
        self.reference_image.resize(height, width)
        for annotation in self.annotations:
            annotation.resize(height, width)
        for so_data_ in self.so_data:
            so_data_.resize(height, width)

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

    def store(self, directory):
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
        self.reference_image.store(directory)
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
            annotation.store(directory)
            annotation_ids.append(
                {
                    'id': str(annotation._id),
                    'type': annotation.get_type()
                }
            )
        f_dict['annotations'] = annotation_ids
        so_data_ids = []
        for so_data in self.so_data:
            so_data.store(directory)
            so_data_ids.append({
                'id': str(so_data._id),
                'type': so_data.get_type()
                }
            )
        f_dict['so_datas'] = so_data_ids
            
        print(f_dict)
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
    def load(cls, directory: str, loader: Optional[SpatialDataLoader] = None):
        if not exists(directory):
            # TODO: Throw custom error message
            pass
        if loader is None:
            loader = SpatialDataLoader.load_default_loader()
        with open(join(directory, 'attributes.json')) as f:
            attributes = json.load(f)
        ref_image_dict = attributes['reference_image']
        ref_image_path = join(directory, ref_image_dict['id'])
        reference_image = loader.load(ref_image_dict['type'], ref_image_path)
        if 'meta_information_path' in attributes:
            meta_information_path = join(directory, attributes['meta_information_path'])
            with open(meta_information_path) as f:
                meta_information = json.load(f)
        else:
            meta_information = None
        annotations = []
        for annotation_dict in attributes['annotations']:
            sub_dir = annotation_dict['id']
            annotation = loader.load(annotation_dict['type'], join(directory, sub_dir))
            annotations.append(annotation)
        so_datas = []
        for so_dict in attributes['so_datas']:
            sub_dir = so_dict['id']
            so_data = loader.load(so_dict['type'],
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
            if isinstance(annotation, Pointset):
                annotation.flip(self.reference_image.data.shape[:2], axis=axis)
            else:
                annotation.flip(axis=axis)
        for so_data_ in self.so_data:
            so_data_.flip(axis=axis)

    @classmethod
    def from_config(cls, config: Dict[str, str]):
        name = config['name']
        image_path = config['image_path']
        image = DefaultImage(data=io.imread(image_path))
        _id = int(config['id'])
        annotations = []
        if 'annotations' in config:
            for path in config['annotations']:
                annotation_data = sitk.GetArrayFromImage(sitk.ReadImage(path))
                # Currently axis need to be swaped due to the way that QuPath exports annotations. (Could also fix this in preprocessing of Annotations.)
                # TODO: Change preprocessing from Qupath output and remove line below.
                if len(annotation_data.shape) > 2:
                    annotation_data = np.moveaxis(annotation_data, 0, -1)
                annotation = Annotation(data=annotation_data)
                annotations.append(annotation)
        if 'segmentation_mask_path' in config:
            segmenation_mask_path = config['segmentation_mask_path']
            segmenation_mask = Annotation(data=io.imread(segmenation_mask_path), name='tissue_mask')
            annotations.append(segmenation_mask)
        if 'landmarks_path' in config:
            landmarks = Pointset(data=pd.read_csv(config['landmarks_path']), name='landmarks')
            annotations.append(landmarks)
        # molecular_data = None
        if 'named_annotations' in config:
            for na_config in config['named_annotations']:
                path = na_config['annotation_path']
                annotation_data = sitk.GetArrayFromImage(sitk.ReadImage(path))
                # Currently axis need to be swaped due to the way that QuPath exports annotations. (Could also fix this in preprocessing of Annotations.)
                # TODO: Change preprocessing from Qupath output and remove line below.
                if len(annotation_data.shape) > 2:
                    annotation_data = np.moveaxis(annotation_data, 0, -1)
                label_path = na_config['label_path']
                with open(label_path, 'r') as f:
                    labels = [x.strip() for x in f.readlines()]
                annotation = Annotation(data=annotation_data, labels=labels)
                annotations.append(annotation)                
        so_datas = []
        if 'so_data' in config:
            for so_config in config['so_datas']:
                so_data = load_spatial_omics_data(so_config['molecular_imaging_data'])
                so_datas.append(so_data)

        obj = cls(image=image,
                   name=name,
                   annotations=annotations,
                   so_data=so_datas,
                   config=config)
        obj._id = _id