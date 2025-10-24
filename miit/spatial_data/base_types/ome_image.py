from bioio_ome_tiff.writers import OmeTiffWriter
from ome_types import to_xml
from bioio_base.types import PhysicalPixelSizes
from lxml import etree

from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
import functools
import json
import os
from os.path import join, exists
import numpy, numpy as np
from typing import Any, Callable
import uuid

import cv2
from bioio import BioImage
import tifffile

from greedyfhist.utils.io import write_to_ometiffile, get_metadata_from_tif
from greedyfhist.data_types.image import (
    INTERPOLATION_TYPE
)
from miit.spatial_data.base_types.image import Image
from miit.spatial_data.base_classes import MIITobject
from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.utils.distance_unit import DUnit
from miit.utils.image_utils import (
    pad,
    crop
)
from miit.utils.utils import create_if_not_exists


def update_interpolation_modes_with_single_value(mode_dict: OrderedDict[str, OrderedDict[str, INTERPOLATION_TYPE]],
                                                 value: INTERPOLATION_TYPE):
    for page in mode_dict:
        for channel in mode_dict[page]:
            mode_dict[page][channel] = value
    return mode_dict


def update_interpolation_modes_with_dict(mode_dict: OrderedDict[str, OrderedDict[str, INTERPOLATION_TYPE]], 
                                         values_dict: dict[str, INTERPOLATION_TYPE | list[INTERPOLATION_TYPE] | dict[str, INTERPOLATION_TYPE]]):
    # Assume that values is as long as the number of pages

    # Raise exception later
    for page_key in values_dict:
        if page_key not in mode_dict:
            raise SetInterpolationException(f'Unknown page: {page_key}')
        page_value = values_dict[page_key]
        if isinstance(page_value, INTERPOLATION_TYPE):
            for channel in mode_dict[page_key]:
                mode_dict[page_key][channel] = page_value
        elif isinstance(page_value, dict):
            for channel_key in page_value:
                if channel_key not in mode_dict[page_key]:
                    raise SetInterpolationException(f'Uknown channel name: {channel_key} for page: {page_key}.')
                mode_dict[page_key][channel_key] = page_value[channel_key]
        elif isinstance(page_value, list):
            if len(page_value) != mode_dict[page_key]:
                raise SetInterpolationException(f'Number of supplied channel values {len(page_value)} does not match channel valyes {len(mode_dict[page_key])} for page {page_key}.')
            for idx, channel_name in enumerate(mode_dict[page_key]):
                mode_dict[page_key][channel_name] = page_value[idx]
        else:
            raise SetInterpolationException(f'Uknown type supplied for page {page_key}: {type(values_dict[page_key])}.')
    return mode_dict
                
            
def update_interpolation_modes_with_list(mode_dict: OrderedDict[str, OrderedDict[str, INTERPOLATION_TYPE]],
                                         values_list: list[INTERPOLATION_TYPE | list[INTERPOLATION_TYPE]]):
    if len(mode_dict) != values_list:
        # Raise exception
        pass
    for page_idx, page_key in enumerate(mode_dict):
        values = values_list[page_idx]
        if isinstance(values, INTERPOLATION_TYPE):
            for channel_name in mode_dict[page_key]:
                mode_dict[page_key][channel_name] = values
        elif isinstance(values, list):
            for channel_idx, channel_name in enumerate(mode_dict[page_key]):
                mode_dict[page_key][channel_name] = values[channel_idx]
        else:
            raise SetInterpolationException(f'Uknown type supplied for page {page_key}: {type(values[page_key])}.')
    return mode_dict
            
      
class SetInterpolationException(Exception):
    pass
            
            
def default_interpolation_ordered_dict():
    return OrderedDict()


def reset_scene_after_use(_func=None, *, var_name='img_data'):
    def decorator_reset_scene_after_use(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not var_name in self.__dict__ and not isinstance(self.__getattribute__(var_name), BioImage):
                raise AttributeError('Decorator cannot be applied on this object.')
            img_data: BioImage = self.__getattribute__(var_name)
            current_scene = img_data.current_scene
            res = func(self, *args, **kwargs)
            img_data.set_scene(current_scene)
            return res
        return wrapper
    
    if _func is None:
        return decorator_reset_scene_after_use
    else:
        return decorator_reset_scene_after_use(_func)


EXTRA_TIF_TAGS: dict[str, int] = {
    'miit_id': 55500,
    'miit_name': 55501,
    'miit_dim_order': 55502 
}


# It should be that pages need to have the same resolution, otherwise we only use the first page.
@MIITobject
@dataclass(kw_only=True)
class OMEImage(Image):
    """
    Class for processing TIFF and OMETIFF images. 
    
    
    Attributes
    ----------
    
    data: numpy.ndarray
        Image data as a numpy array. 
        
    path: str
        Original file path.
        
    tif: tifffile.tifffile.TiffFile 
        Connection to tif image information.

    is_ome: bool
        Indicates whether the read file is ome.
    """
    img: BioImage
    tif_metadata: dict = field(default_factory=lambda: defaultdict(dict))
    main_page: int | str = 0
    main_channel: int | str = 0
    dim_order: str = ''
    path: str | os.PathLike = ''
    interpolation_mode: OrderedDict[str, OrderedDict[str, INTERPOLATION_TYPE]]  = field(default_factory=default_interpolation_ordered_dict) # type: ignore
    active_pages: list[str] = field(default_factory=lambda: [])
    
    @property
    def data(self) -> numpy.ndarray:
        # Get channel idx
        if isinstance(self.main_channel , str):
            channel_names = self.img.get_channel_names()
            channel_idx = channel_names.index(self.main_channel)
        else:
            channel_idx = self.main_channel
        data = self.img.get_image_data(self.dim_order, C=channel_idx)
        return data
    
    @data.setter
    def data(self, data: numpy.ndarray):
        pass
    
    @reset_scene_after_use(var_name='img')
    def __post_init__(self):
        super().__post_init__()
        if self.tif_metadata is not None:
            x_size = self.tif_metadata.get('PhysicalSizeX', 1.0)
            x_unit = self.tif_metadata.get('PhysicalSizeXUnit', 'pixel')
            y_size = self.tif_metadata.get('PhysicalSizeY', 1.0)
            y_unit = self.tif_metadata.get('PhysicalSizeYUnit', 'pixel')
            self.resolution = (DUnit(x_size, x_unit), DUnit(y_size, y_unit))
        if self.dim_order == '':
            self.dim_order = 'YXS' if 'S' in self.img.dims.order else 'YX'
        if not self.interpolation_mode:
            self.__init_interpolation_mode__()
        # Do an integrity check for spacing
        self.img.set_scene(self.main_page)
        main_x = self.img.dims['X'][0]
        main_y = self.img.dims['Y'][0]
        active_pages = []
        for page in self.img.scenes:
            self.img.set_scene(page)
            if main_x == self.img.dims['X'][0] and main_y == self.img.dims['Y'][0]:
                active_pages.append(page)
        self.active_pages = active_pages
                
    @reset_scene_after_use(var_name='img')
    def __init_interpolation_mode__(self, default_interpolation_mode: str | int = 'LINEAR'):
        int_modes: OrderedDict[str, OrderedDict[str, INTERPOLATION_TYPE]] = OrderedDict()
        for scene in self.img.scenes:
            int_modes[scene] = OrderedDict()
            self.img.set_scene(scene)
            for channel_name in self.img.channel_names:
                int_modes[scene][channel_name] = default_interpolation_mode
        self.interpolation_mode = int_modes
                
    def set_interpolation_mode(self, modes: INTERPOLATION_TYPE | list | dict):
        if isinstance(modes, INTERPOLATION_TYPE):
            int_modes = update_interpolation_modes_with_single_value(self.interpolation_mode, modes)
        elif isinstance(modes, list):
            int_modes = update_interpolation_modes_with_list(self.interpolation_mode, modes)
        elif isinstance(modes, dict):
            int_modes = update_interpolation_modes_with_dict(self.interpolation_mode, modes)
        else:
            raise SetInterpolationException(f'No function to pass type of interpolation modes: {type(modes)}')
        self.interpolation_mode = int_modes                 # type: ignore

    # TODO: What is the difference between the two.
    def get_resolution(self) -> tuple[DUnit, DUnit]:
        return self.resolution

    @reset_scene_after_use(var_name='img')
    def apply_transform(self, 
                       registerer: Registerer, 
                       transformation: RegistrationResult | numpy.ndarray, 
                       **kwargs: dict) -> Any:
        current_scene = self.img.current_scene
        transformed_data_outer = []
        all_channel_names = []
        dim_order=self.img.dims.order
        physical_pixel_sizes = self.img.physical_pixel_sizes
        for img_idx, page in enumerate(self.img.scenes):
            self.img.set_scene(page)
            transformed_data_inner = []
            channel_names = []
            if page in self.active_pages:
                for channel_idx, channel_name in enumerate(self.img.channel_names):
                    channel_names.append(channel_name)
                    img_data = self.img.get_image_data(self.dim_order, C=channel_idx)
                    interpolation = self.interpolation_mode[page][channel_name]
                    transformed_data = registerer.transform_image(img_data, 
                                                                transformation, 
                                                                interpolation,
                                                                **kwargs)
                    # Very hacky. There is probably a better to do that.
                    r = np.array([True]*len(self.img.dims.order))
                    for d in self.dim_order:
                        r[self.img.dims.order.find(d)] = False
                    dims_to_expand = np.where(r)[0].tolist()
                    transformed_data = np.expand_dims(transformed_data, axis=dims_to_expand)
                    transformed_data_inner.append(transformed_data)
                all_channel_names.append(channel_names)
                axis_idx = self.img.dims.order.index('C')
                transformed_data_inner = np.concatenate(transformed_data_inner, axis=axis_idx)
            else:
                transformed_data_inner = self.img.data
                all_channel_names.append(self.img.channel_names)
            transformed_data_outer.append(transformed_data_inner)        
        self.img.set_scene(current_scene)
        transformed_bioio_image = BioImage(image=transformed_data_outer,
                                           dim_order=dim_order,
                                           channel_names=all_channel_names,
                                           physical_pixel_sizes=physical_pixel_sizes)
        # Now do some postprocessing, since BioImage doesnt provide some the functions we need.
        # 1. Add proper scene names
        if len(transformed_bioio_image.reader.scenes) != len(self.img.scenes):
            # Raise exception in the future
            print('Raise exception in the future.')
            pass
        transformed_bioio_image.reader._scenes = self.img.scenes
        return OMEImage(
            img=transformed_bioio_image,
            tif_metadata=self.tif_metadata.copy(),
            name=self.name,
            main_page=self.main_page,
            main_channel=self.main_channel,
            dim_order=self.dim_order,
            path=self.path,
            interpolation_mode=self.interpolation_mode
        )
        
    @reset_scene_after_use(var_name='img')
    def __apply_transform_along_channels(self,
                                         fun: Callable[[numpy.ndarray], numpy.ndarray]) -> BioImage:
        current_scene = self.img.current_scene
        transformed_data_outer = []
        all_channel_names = []
        for img_idx, page in enumerate(self.img.scenes):
            if page in self.active_pages:
                self.img.set_scene(page)
                transformed_data_inner = []
                channel_names = []
                for channel_idx, channel_name in enumerate(self.img.channel_names):
                    channel_names.append(channel_name)
                    img_data = self.img.get_image_data(self.dim_order, C=channel_idx)
                    transformed_data = fun(img_data)
                    # Very hacky. There is probably a better to do that.
                    r = np.array([True]*len(self.img.dims.order))
                    for d in self.dim_order:
                        r[self.img.dims.order.find(d)] = False
                    dims_to_expand = np.where(r)[0].tolist()
                    transformed_data = np.expand_dims(transformed_data, axis=dims_to_expand)
                    transformed_data_inner.append(transformed_data)
            else:
                channel_names = self.img.channel_names
                transformed_data_inner = self.img.data
            all_channel_names.append(channel_names)
            axis_idx = self.img.dims.order.index('C')
            transformed_data_inner = np.concatenate(transformed_data_inner, axis=axis_idx)
            transformed_data_outer.append(transformed_data_inner)        
        self.img.set_scene(current_scene)
        transformed_bioio_image = BioImage(image=transformed_data_outer,
                                           dim_order=self.img.dims.order,
                                           channel_names=all_channel_names,
                                           physical_pixel_sizes=self.img.physical_pixel_sizes)
        return transformed_bioio_image
    
    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        fun = lambda data: crop(data, xmin, xmax, ymin, ymax)
        new_img = self.__apply_transform_along_channels(fun)
        self.img = new_img
        
    def pad(self, padding: tuple[int, int, int, int], constant_values: int = 0):
        fun = lambda data: pad(data, padding, constant_values)
        new_img = self.__apply_transform_along_channels(fun)
        self.img = new_img

    def resize(self, width: int, height: int):
        # Use opencv's resize function here, because it typically works a lot faster and for now
        # we assume that data in Image is always some kind of rgb like image.
        w, h = self.data.shape[:2]
        w_rate = w / width
        h_rate = h / height
        fun = lambda data: cv2.resize(data, (height, width))
        new_img = self.__apply_transform_along_channels(fun)
        self.img = new_img
        self.scale_resolution((w_rate, h_rate))

    def rescale(self, scaling_factor: float | tuple[float, float]):
        if not isinstance(scaling_factor, tuple):
            scaling_factor = (scaling_factor, scaling_factor)
        w, h = self.data.shape[:2]
        w_n, h_n = int(w*scaling_factor[0]), int(h*scaling_factor[1])
        self.resize(w_n, h_n)

    @staticmethod
    def clean_unit_symbol(symbol: str):
        # Check for more custom types
        if symbol == 'px':
            symbol = 'pixel'
        if symbol == 'um':
            symbol = 'µm'
        return symbol

    @reset_scene_after_use(var_name='img')
    def store_as_ome_tiff_image(self, 
                                path: str,
                                save_id: bool = True,
                                save_name: bool = True,
                                save_dimorder: bool = True):
        # We use ome tiff, but bioimage and ome_types dont encode
        # PhysicalUnitSizes. So we first create the ome xml, add
        # the the UnitSizes and then write the image to file.
        data = []
        scenes = []
        all_channels = []
        dim_orders = []

        for scene in self.img.scenes:
            self.img.set_scene(scene)
            data.append(self.img.data)
            scenes.append(scene)
            all_channels.append(self.img.channel_names)
            dim_orders.append(self.img.dims.order)

        image_name = [self.name] * len(data)
        channel_colors = None
        
        # We assume that physical sizes are consistent across all pages in a tiff file.
        physical_pixel_sizes = PhysicalPixelSizes(Z=0,
                                                X=float(self.resolution[0].value),
                                                Y=float(self.resolution[1].value))

        physical_pixel_sizes = [physical_pixel_sizes] * len(data)

        ome_xml = OmeTiffWriter.build_ome(
            [i.shape for i in data],
            [i.dtype for i in data],
            channel_names=all_channels,  # type: ignore
            image_name=image_name, # type: ignore
            physical_pixel_sizes=physical_pixel_sizes,
            channel_colors=channel_colors,  # type: ignore
            dimension_order=dim_orders,
        )
        ome_xml = to_xml(ome_xml)
        
        root = etree.fromstring(ome_xml) # type: ignore
        pixels = root.findall('Image/Pixels', namespaces=root.nsmap)
        for pixel in pixels:
            x_unit = OMEImage.clean_unit_symbol(self.resolution[0].symbol)
            y_unit = OMEImage.clean_unit_symbol(self.resolution[1].symbol)
            pixel.attrib['PhysicalSizeXUnit'] = x_unit
            pixel.attrib['PhysicalSizeYUnit'] = y_unit
        updated_ome_xml = etree.tostring(root, encoding="unicode") # type: ignore
        
        extra_tags = []
        # Get reverse tif tags
        if save_id:
            tpl = (EXTRA_TIF_TAGS['miit_id'], 's', 0, str(self._id), True)
            extra_tags.append(tpl)
        if save_name and self.name is not None:
            tpl = (EXTRA_TIF_TAGS['miit_name'], 's', 0, self.name, True)
            extra_tags.append(tpl)
        if save_dimorder:
            tpl = (EXTRA_TIF_TAGS['miit_dim_order'], 's', 0, self.dim_order, True)
            extra_tags.append(tpl)
        if not extra_tags:
            extra_tags = None
            
        
        OmeTiffWriter.save(
            data=data,
            uri=path,
            dim_order=dim_orders,
            ome_xml=updated_ome_xml,
            channel_names=all_channels,
            tifffile_kwargs={
                'extratags': extra_tags
            }
        )

    def store(self, 
              path: str):
        # THIS NEEDS TO BE FIXED AS WELL
        create_if_not_exists(path)
        fname = 'image.ome.tif'
        image_path = join(path, fname)
        self.store_as_ome_tiff_image(
            image_path,
            save_id=True,
            save_name=True
        )
        if self.meta_information:
            additional_attributes = {
                'meta_information': self.meta_information
            }
            with open(join(path, 'additional_attributes.json'), 'w') as f:
                json.dump(additional_attributes, f)

    # Will be deprecated
    def to_file__(self, path: str):
        write_to_ometiffile(
            self.data, path, self.tif_metadata, False
        )

    @staticmethod
    def get_type() -> str:
        return 'OMEImage'       

    @classmethod
    def load(cls, path: str) -> 'OMEImage':
        aa_path = join(path, 'additional_attributes.json')
        if exists(aa_path):
            with open(aa_path) as f:
                additional_attributes = json.load(f) 
        else:
            additional_attributes = {
            }
        meta_information = additional_attributes.get('meta_information', {})
        image_path = join(path, 'image.ome.tif')
        # Get extra tags
        obj = OMEImage.load_from_path(image_path)
        obj.meta_information = meta_information
        return obj
           
    @classmethod
    def load_from_path(cls, 
                       path: str,
                       name: str = '',
                       dim_order: str = '') -> 'OMEImage':
        img = BioImage(path)
        if not name:
            name = os.path.basename(path)
        # 1. Since bioio doesnt store physical size units, we need to open the file again and get it out ourselves.
        id_ = None
        if os.path.isfile(path):
            with tifffile.TiffFile(path) as tif:
                if tif.is_ome:
                    tif_metadata = get_metadata_from_tif(tif.ome_metadata) # type: ignore
                else:
                    tif_metadata = {
                        'PhysicalSizeXUnit': 'pixel',
                        'PhysicalSizeYUnit': 'pixel'
                    }
                page0: tifffile.tifffile.TiffPage = tif.pages[0] # type: ignore
                if EXTRA_TIF_TAGS['miit_id'] in page0.tags:
                    id_ = uuid.UUID(page0.tags[EXTRA_TIF_TAGS['miit_id']].value)
                if EXTRA_TIF_TAGS['miit_name'] in page0.tags:
                    name = page0.tags[EXTRA_TIF_TAGS['miit_name']].value
                if EXTRA_TIF_TAGS['miit_dim_order'] in page0.tags:
                    dim_order = page0.tags[EXTRA_TIF_TAGS['miit_dim_order']].value
        else:
            tif_metadata = {
                'PhysicalSizeXUnit': 'pixel',
                'PhysicalSizeYUnit': 'pixel'
            }
            
        obj = cls(img=img, 
                  name=name, 
                  tif_metadata=tif_metadata,
                  dim_order=dim_order)
        if id_ is not None:
            obj._id = id_
        return obj