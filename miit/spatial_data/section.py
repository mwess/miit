from dataclasses import dataclass, field
import json
import os
from os.path import join, exists
from typing import Any
import uuid
from uuid import UUID
import shutil

import numpy, numpy as np

from miit.spatial_data.base_classes import (
    BaseImage,
    BasePointset,
    BaseSpatialOmics,
    ImagingDataIO,
    IMAGING_DATA_IO
)

from miit.spatial_data.base_types import (
    Image
)

from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.registerers.opencv_affine_registerer import OpenCVAffineRegisterer
from miit.spatial_data.base_types.geojson import GeoJSONData
from miit.utils.utils import copy_if_not_none, get_half_pad_size

# TODO: Refactor this out. This can probably be easily replaced using OpenCV.
def get_boundary_box(image: numpy.ndarray, background_value: float = 0) -> tuple[int, int, int, int]:
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
    for layer in section.layers:
        name_len = len(layer.name)
        if max_name_len < name_len:
            max_name_len = name_len
        id_len = len(str(layer._id))
        if max_id_len < id_len:
            max_id_len = id_len
        type_len = len(layer.get_type())
        if max_type_len < type_len:
            max_type_len = type_len
        identifiers.append((layer.name, str(layer._id), layer.get_type()))

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


def groupwise_registration(sections: list['Section'],
                           registerer: Registerer,
                           **kwargs: dict) -> tuple[list['Section'], list[RegistrationResult]]:
    """
    Performs a groupwise registration on all supplied sections. A registration between all sections is computed
    and applied to all sections. Per convention, the last section in sections is used as the fixed section.
    
    Another convention: If custom masks are to be used, they need to be supplied in an annotation called tissue_mask.
    
    Args:
        sections (List[Section]):
        registerer (Registerer):

    Returns:
        tuple[list[Section], list[RegistrationResult]]
    """
    # TODO: Allow to add options to the registerer.
    # TODO: Not all registerer can do groupwise registration. Find a way to filter those out.
    # Setup list with images
    images_with_mask_list = []
    for idx, section in enumerate(sections):
        if section.reference_image is None:
            raise Exception('Section misses a reference image.')
        image = section.reference_image.data
        # mask_annotation = section.segmentation_mask
        mask_annotation = section.get_annotation_by_name('tissue_mask')
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


def register_to_ref_image(target_image: numpy.ndarray | BaseImage, 
                          source_image: numpy.ndarray | BaseImage, 
                          data: BaseImage | BasePointset | BaseSpatialOmics,
                          registerer: Registerer | None = None,
                          reg_opts: dict | None = None,
                          **args) -> tuple[BaseImage | BasePointset | BaseSpatialOmics, RegistrationResult, Image]:
    """
    Utility function to register some spatial data to a target image space.
    Registers the source to the target image and applies the transformation to additionally supplied data.

    Args:
        target_image (numpy.ndarray | BaseImage):
        source_image (numpy.ndarray | BaseImage):
        data (BaseImage | BasePointset): Additional data  to transform. 
        registerer (Registerer, optional): If None, uses the OpenCVAffineRegisterer. Defaults to None.
        reg_opts (dict | None, optional): Options parsed to the registerer.. Defaults to None.

    Returns:
        tuple[BaseImage | BasePointset, RegistrationResult, Image]: Tuple of warped data, computed transformation, and warped source image
    """
    if isinstance(target_image, BaseImage):
        target_image = target_image.data
    if isinstance(source_image, BaseImage):
        source_image = source_image.data
    if registerer is None:
        registerer = OpenCVAffineRegisterer()
    if reg_opts is None:
        reg_opts = {}
    transformation: RegistrationResult = registerer.register_images(source_image, target_image, **reg_opts)
    warped_data = data.apply_transform(registerer, transformation)
    warped_source_image = Image(data=source_image).apply_transform(registerer, transformation)
    return warped_data, transformation, warped_source_image


@dataclass(init=False)
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
        
    layers: List[Union[BaseImage, BasePointset]] = field(default_factory=lambda: [])
        List of additional layers. 
    
    meta_information: Optional[Dict[Any, Any]] = None
        Additional meta information.
    """
    __ref_img_idx__: int | None = field(init=False, repr=False)
    name: str | None = None
    _id: uuid.UUID = field(init=False) 
    layers: list[BaseImage | BasePointset | BaseSpatialOmics] = field(default_factory=lambda: [])
    meta_information: dict[Any, Any] | None = None

    def __init__(self,
                reference_image: int | str | UUID | BaseImage | BaseSpatialOmics | None = None,
                name: str | None = None,
                layers: list[BaseImage | BasePointset | BaseSpatialOmics] | None = None,
                meta_information: dict[Any, Any] | None = None,
                id_: uuid.UUID | str | None = None):
        self.name = name
        if id_ is None:
            self._id = uuid.uuid1()
        elif isinstance(id_, str):
            self._id = uuid.UUID(id_)
        else:
            self._id = id_
        if layers is None:
            layers = []
        self.layers = layers
        self.meta_information = meta_information
        if reference_image is not None:
            self.reference_image = reference_image

    @property
    def reference_image(self):
        if self.__ref_img_idx__ is not None:
            return self.layers[self.__ref_img_idx__]

    @reference_image.setter
    def reference_image(self, 
                        ref_image: int | str | UUID | BaseImage | BaseSpatialOmics):
        """Setter method for reference image. 

        Args:
        
            ref_image (int | str | UUID | BaseImage | BaseSpatialOmics):
                If `ref_image` is an int, `ref_image` is treated as an index for the spatial objects in `layers`. If `ref_image` is 
                out-of-bounds, nothing happens.

                If `ref_image` is a str or UUID, the spatial data with the corresponding `id` is searched in `layers`. If `ref_image` 
                canot be found, nothing happens. 

                If `ref_image` is a `BaseImage` or `BaseSpatialOmics`, the behaviour depends on whether `ref_image` is 
                already in `layers`. If `ref_image` already exists, no new image will be added and `reference_image` will return
                the existing spatial object. Otherwise, `ref_image` will be added to `layers`.

        """
        if isinstance(ref_image, int):
            if len(self.layers) > ref_image:
                self.__ref_img_idx__ = ref_image
        elif isinstance(ref_image, UUID):
            ref_image = str(ref_image)
        elif isinstance(ref_image, str):
            layer_idx = self.find_layer_index_by_name(ref_image)
            if layer_idx is not None:
                self.__ref_img_idx__ = layer_idx
        elif isinstance(ref_image, BaseImage) or isinstance(ref_image, BaseSpatialOmics):
            # First find out, if we already have this image.
            layer_idx = self.find_layer_index_by_id(ref_image._id)
            if layer_idx is not None:
                self.__ref_img_idx__ = ref_image
            else:
                self.layers.append(ref_image)
                self.__ref_img_idx__ = len(self.layers) - 1
        else:
            raise Exception('Tried to set an unknown data type.')
        
    def __hash__(self) -> int:
        return self._id

    def copy(self) -> 'Section':
        """Returns a copy of a section.

        Returns:
            Section:
        """
        config = copy_if_not_none(self.meta_information)
        layers = self.layers.copy()
        ref_img_idx = self.__ref_img_idx__ if self.__ref_img_idx__ is not None else None
        copied_section = Section(
            reference_image=ref_img_idx,
            name=self.name,
            layers=layers,
            meta_information=config)
        copied_section._id = self._id
        return copied_section

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        """Applies cropping to each datatype within the section.

        Args:
            xmin (int):
            xmax (int):
            ymin (int):
            ymax (int):
        """
        for layer in self.layers:
            layer.crop(xmin, xmax, ymin, ymax)

    def crop_by_mask(self, mask: numpy.ndarray):
        """Crops every datatype within the section based on the mask boundaries.

        Args:
            mask (numpy.ndarray):
        """
        xmin, xmax, ymin, ymax = get_boundary_box(mask)
        self.crop(xmin, xmax, ymin, ymax)

    def pad(self, padding: tuple[int, int, int, int]):
        """Applies `padding` to each datatype within a Section.

        Args:
            padding (tuple[int, int, int, int]):
        """
        for layer in self.layers:
            layer.pad(padding)

    def resize(self, width: int, height: int):
        """Applies `resize` to each datatype within a Section.

        Args:
            width (int): 
            height (int):
        """
        if self.__ref_img_idx__ is None:
            does_reference_image_exist = False
        else:
            does_reference_image_exist = True
            w, h = self.reference_image.data.shape[:2]
            ws = width / w
            hs = height / h
        for layer in self.layers:
            if isinstance(layer, BasePointset):
                if not does_reference_image_exist:
                    raise Exception('BasePointsets cannot be transformed, since no reference image exists.')
                layer.resize(ws, hs)
            else:
                layer.resize(width, height)

    # TODO: Implement function
    def rescale(self, scaling_factor: float):
        pass

    def apply_transform(self, 
             registerer: Registerer, 
             transformation: RegistrationResult, 
             **kwargs: dict) -> 'Section':
        """Applies transformation to all spatially resolved data in the section object.
        Registerer and transformation need to be from the same registration algorithm.

        Args:
            registerer (Registerer):
            transformation (RegistrationResult):

        Returns:
            Section: Wapred section.
        """
        layers_transformed = []
        for layer in self.layers:
            layer_transformed = layer.apply_transform(registerer, transformation, **kwargs)
            layers_transformed.append(layer_transformed)
        meta_information = self.meta_information.copy() if self.meta_information is not None else None
        transformed_section = Section(reference_image=self.__ref_img_idx__,
                       name=self.name,
                       layers=layers_transformed,
                       meta_information=meta_information)
        transformed_section._id = self._id
        return transformed_section

    def store(self, 
              directory: str,
              imaging_data_io: ImagingDataIO | None = None):
        # TODO: Rewrite that function.
        """
        Should look like this:
            File with mapping what attribute in what subfolder.
            Each subfolder is marked with their id.
        """
        if imaging_data_io is None:
            imaging_data_io = IMAGING_DATA_IO
        if exists(directory):
            shutil.rmtree(directory)
        if not exists(directory):
            os.mkdir(directory)
        f_dict = {}
        f_dict['id'] = str(self._id)
        f_dict['name'] = self.name
        if self.__ref_img_idx__ is not None:
            f_dict['ref_img_idx'] = self.__ref_img_idx__
        meta_information_path = join(directory, 'meta_information.json')
        with open(meta_information_path, 'w') as f:
            json.dump(self.meta_information, f)
            f_dict['meta_information_path'] = 'meta_information.json' 
        layer_ids = []
        for layer in self.layers:
            layer.store(join(directory, str(layer._id)))
            layer_ids.append(
                {
                    'id': str(layer._id),
                    'type': layer.get_type()
                }
            )
        f_dict['layers'] = layer_ids
            
        with open(join(directory, 'attributes.json'), 'w') as f:
            json.dump(f_dict, f)

    def find_layer_index_by_name(self, name: str) -> int | None:
        """Find the index of the layer based on the name.

        Args:
            name (str):

        Returns:
            int | None: Index of layer or None if image cannot be found.
        """
        for idx, layer in enumerate(self.layers):
            if not hasattr(layer, 'name'):
                continue
            if layer.name == name:
                return idx
        return None

    def get_annotation_by_name(self, name: str) -> BaseImage | BasePointset | BaseSpatialOmics | None:
        """Finds an annotation by matching Annotation.name with name.
        Returns None, if annotation could not be found.

        Args:
            name (str):

        Returns:
            BaseImage | None:
        """
        layer_idx = self.find_layer_index_by_name(name)
        if layer_idx is None:
            return None
        return self.layers[layer_idx]
    
    def find_layer_index_by_id(self, id_: str | UUID) -> int | None:
        """Find the index of the layer based on the name.

        Args:
            id_ (str | UUID):

        Returns:
            int | None: Index of layer or None if image cannot be found.
        """
        if isinstance(id_, UUID):
            id_ = str(id_)
        for idx, layer in enumerate(self.layers):
            if str(layer._id) == id_:
                return idx
        return None

    def get_annotation_by_id(self, id_: str) -> BaseImage | BasePointset | BaseSpatialOmics | None:
        """Gets annotation by matching Annotation._id with id_.
        Returns None, if annotation can not be found.

        Args:
            id_ (str): 

        Returns:
            BaseImage | None: 
        """
        layer_idx = self.find_layer_index_by_id(id_)
        if layer_idx is None:
            return None
        return self.layers[layer_idx]
    
    def print_data_summary(self):
        """Prints a summary of data.
        """
        print(get_table_summary_string(self))
            
    @classmethod
    def load(cls, directory: str, 
             imaging_data_io: ImagingDataIO | None = None) -> 'Section':
        """Loads a Section object from directory. 

        Args:
            directory (str): Source directory.
            imaging_data_io (Optional[ImagingDataIO], optional): Data loader for imaging data. 
                If None, uses a default ImagingDataIO. Defaults to None.

        Returns:
            Section: 
        """
        if not exists(directory):
            # TODO: Throw custom error message
            pass
        if imaging_data_io is None:
            imaging_data_io = IMAGING_DATA_IO
        with open(join(directory, 'attributes.json')) as f:
            attributes = json.load(f)
        reference_image = attributes.get('ref_img_idx', None)
        if 'meta_information_path' in attributes:
            meta_information_path = join(directory, attributes['meta_information_path'])
            with open(meta_information_path) as f:
                meta_information = json.load(f)
        else:
            meta_information = None
        layers = []
        for layer_dict in attributes['layers']:
            sub_dir = layer_dict['id']
            layer = imaging_data_io.load(layer_dict['type'], 
                                         join(directory, sub_dir))
            layers.append(layer)
        obj = cls(
            reference_image=reference_image,
            name=attributes['name'],
            layers=layers,
            meta_information=meta_information
        )
        obj._id = uuid.UUID(attributes['id'])
        return obj

    def add_molecular_imaging_data(self,
                                   mi: BaseSpatialOmics,
                                   do_register_to_ref_image=True,
                                   reference_image: numpy.ndarray = None,
                                   registerer: Registerer | None = None):
        """
        Adds molecular imaging data to section.
        
        If register_to_primary_image is True, registers the molecular imaging data to the primary image.
        reference_image: Needs to be set it molecular imaging data is to be registered to the primary image.
        
        Args:
            mi (BaseSpatialOmics): _description_
            register_to_primary_image (bool, optional): _description_. Defaults to True.
            reference_image (numpy.ndarray, optional): _description_. Defaults to None.
            registerer (Registerer | None, optional): _description_. Defaults to None.
        """

        # TODO: Add this function to loading from config!
        if self.__ref_img_idx__ is None:
            raise Exception('add_molecular_imaging_data requires a reference image')
        if do_register_to_ref_image:
            warped_mi, _, _ = register_to_ref_image(self.reference_image.data, reference_image, mi, registerer)
        else:
            warped_mi = mi
        self.layers.append(warped_mi)

    def flip(self, axis: int = 0):
        """
        Flip all spatial data in this section.
        """
        for layer in self.layers:
            if isinstance(layer, BasePointset):
                layer.flip(self.reference_image.data.shape[:2], axis=axis)
            else:
                layer.flip(axis=axis)