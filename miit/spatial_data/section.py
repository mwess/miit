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

from miit.spatial_data.molecular_imaging.imaging_data import BaseMolecularImaging
from miit.spatial_data.molecular_imaging.loader import MolecularImagingLoader
from miit.spatial_data.loaders import load_molecular_imaging_data, load_molecular_imaging_data_from_directory
from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.registerers.opencv_affine_registerer import OpenCVAffineRegisterer
from miit.spatial_data.image import BaseImage, Image, Annotation, Pointset
from miit.utils.utils import copy_if_not_none


def get_boundary_box(image: numpy.array, background_value: float =0) -> Tuple[int, int, int, int]:
    points = np.argwhere(image != background_value)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    return xmin, xmax, ymin, ymax


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
        image = section.image.data
        # mask_annotation = section.segmentation_mask
        mask_annotation = section.get_annotations_by_names('tissue_mask')
        mask = mask_annotation.data if mask_annotation is not None else None
        images_with_mask_list.append((image, mask))
    # Now do groupwise registration.
    transforms, _ = registerer.groupwise_registration(images_with_mask_list, skip_deformable_registration=skip_deformable_registration)
    warped_sections = []
    for idx, section in enumerate(sections[:-1]):
        transform = transforms[idx]
        warped_section = section.warp(registerer, transform)
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
    warped_data = data.warp(registerer, transformation)
    warped_ref_image = Image(data=source_image).warp(registerer, transformation).data
    return warped_data, warped_ref_image

@dataclass
class Section:
    image: Image
    name: str
    id_: int
    int_id_: uuid.UUID = field(init=False)
    segmentation_mask: Optional[Annotation] = None
    landmarks: Optional[Pointset] = None
    molecular_data: Optional[BaseMolecularImaging] = None
    molecular_datas: List[BaseMolecularImaging] = field(default_factory=lambda: [])
    annotations: List[Union[Annotation, Pointset]] = field(default_factory=lambda: [])
    config: Optional[Dict[Any, Any]] = None


    def __post_init__(self):
        self.int_id_ = uuid.uuid1()
        self.preprocess_imaging_data()

    def __hash__(self) -> int:
        return self.id_

    def copy(self):
        image = self.image.copy()
        segmentation_mask = copy_if_not_none(self.segmentation_mask)
        landmarks = copy_if_not_none(self.landmarks)
        config = copy_if_not_none(self.config)
        molecular_data = copy_if_not_none(self.molecular_data)
        annotations = self.annotations.copy()
        return Section(
            image=image,
            name=self.name,
            id_=self.id_,
            segmentation_mask=segmentation_mask,
            landmarks=landmarks,
            molecular_data=molecular_data,
            annotations=annotations,
            config=config)

    def preprocess_imaging_data(self):
        if self.config.get('enable_mask_cropping', False):
            self.crop_by_mask()

    def apply_bounding_parameters(self, xmin, xmax, ymin, ymax):
        self.image.apply_bounding_parameters(xmin, xmax, ymin, ymax)
        if self.segmentation_mask is not None:
            self.segmentation_mask.apply_bounding_parameters(xmin, xmax, ymin, ymax)
        if self.landmarks is not None:
            self.landmarks.apply_bounding_parameters(xmin, xmax, ymin, ymax)
        if self.molecular_data is not None:
            self.molecular_data.apply_bounding_parameters(xmin, xmax, ymin, ymax)
        for annotation in self.annotations:
            annotation.apply_bounding_parameters(xmin, xmax, ymin, ymax)

    def crop_by_mask(self, mask):
        # if self.segmentation_mask is None:
        #     print('Cannot do autocropping by segmentation mask as mask does not exist.')
        #     # TODO: Throw exception
        xmin, xmax, ymin, ymax = get_boundary_box(mask)
        self.apply_bounding_parameters(xmin, xmax, ymin, ymax)

    def pad(self, padding: Tuple[int, int, int, int]):
        self.image.pad(padding)
        if self.segmentation_mask is not None:
            self.segmentation_mask.pad(padding)
        if self.landmarks is not None:
            self.landmarks.pad(padding)
        if self.molecular_data is not None:
            self.molecular_data.pad(padding)
        for annotation in self.annotations:
            annotation.pad(padding)

    def rescale_data(self, height, width):
        h_old, w_old = self.image.data.shape[0], self.image.data.shape[1]
        self.image.rescale(height, width)
        width_scale = width / w_old
        height_scale = height / h_old
        if self.segmentation_mask is not None:
            self.segmentation_mask.rescale(height, width)
        if self.landmarks is not None:
            self.landmarks.rescale(height_scale, width_scale)
        if self.molecular_data is not None:
            self.molecular_data.rescale(height, width)

    def warp(self, 
             registerer: Registerer, 
             transformation: RegistrationResult, 
             **kwargs: Dict) -> 'Section':
        image_transformed = self.image.warp(registerer, transformation, **kwargs)
        mask_transformed = None
        if self.segmentation_mask is not None:
            mask_transformed = self.segmentation_mask.warp(registerer, transformation, **kwargs)
        landmarks_transformed = None
        if self.landmarks is not None:
            landmarks_transformed = self.landmarks.warp(registerer, transformation, **kwargs)
        molecular_data_transformed = None
        if self.molecular_data is not None:
            molecular_data_transformed = self.molecular_data.warp(registerer, transformation, **kwargs)
        annotations_transformed = []
        for annotation in self.annotations:
            annotation_transformed = annotation.warp(registerer, transformation, **kwargs)
            annotations_transformed.append(annotation_transformed)
        config = self.config if self.config is not None else None
        return Section(image=image_transformed,
                       name=self.name,
                       id_=self.id_,
                       segmentation_mask=mask_transformed,
                       landmarks=landmarks_transformed,
                       molecular_data=molecular_data_transformed,
                       annotations=annotations_transformed,
                       config=config)

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
        f_dict['id'] = self.id_
        f_dict['int_id'] = str(self.int_id_)
        f_dict['name'] = self.name
        self.image.store(directory)
        f_dict['primary_image'] = str(self.image.id_)
        if self.landmarks is not None:
            self.landmarks.store(directory)
            f_dict['landmarks'] = str(self.landmarks.id_)
        config_path = join(directory, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f)
            f_dict['config_path'] = config_path
        if self.segmentation_mask is not None:
            self.segmentation_mask.store(directory)
            f_dict['segmentation_mask'] = str(self.segmentation_mask.id_)
        if self.molecular_data is not None:
            self.molecular_data.store(directory)
            f_dict['molecular_data'] = str(self.molecular_data.id_)
            f_dict['molecular_data_type'] = self.molecular_data.get_type()
        annotation_ids = []
        for annotation in self.annotations:
            annotation.store(directory)
            annotation_ids.append(str(annotation.id_))
        f_dict['annotations'] = annotation_ids
        with open(join(directory, 'attributes.json'), 'w') as f:
            json.dump(f_dict, f)

    def get_annotations_by_names(self, name: str):
        for annotation in self.annotations:
            if annotation.name == name:
                return annotation
        return None
    
    def get_annotation_by_id(self, id_: str):
        for annotation in self.annotations:
            if str(annotation.id_) == id_:
                return annotation
        return None
            

    @classmethod
    def load(cls, directory: str, loader: Optional[MolecularImagingLoader] = None):
        if not exists(directory):
            # TODO: Throw custom error message
            pass
        if loader is None:
            loader = MolecularImagingLoader.load_default_loader()
        with open(join(directory, 'attributes.json')) as f:
            attributes = json.load(f)
        image = Image.load(join(directory, attributes['primary_image']))
        if 'landmarks' in attributes:
            landmarks = Pointset.load(join(directory, attributes['landmarks']))
        else:
            landmarks = None
        if 'config_path' in attributes:
            with open(attributes['config_path']) as f:
                config = json.load(f)
        else:
            config = None
        if 'segmentation_mask' in attributes:
            segmentation_mask = Annotation.load(join(directory, attributes['segmentation_mask']))
        else:
            segmentation_mask = None
        if 'molecular_data' in attributes:
            molecular_data = loader.load(attributes['molecular_data_type'],
                                                         join(directory, attributes['molecular_data']))
            # molecular_data = load_molecular_imaging_data_from_directory(attributes['molecular_data_type'], 
            #                                                             join(directory, attributes['molecular_data']))
        else:
            molecular_data = None
        annotations = []
        for sub_dir in attributes['annotations']:
            if exists(join(directory, sub_dir, 'pointset.csv')):
                annotation = Pointset.load(join(directory, sub_dir))
            else:
                annotation = Annotation.load(join(directory, sub_dir))
            annotations.append(annotation)
        obj = cls(
            image=image,
            name=attributes['name'],
            segmentation_mask=segmentation_mask,
            landmarks=landmarks,
            molecular_data=molecular_data,
            id_=attributes['id'],
            annotations=annotations,
            config=config
        )
        # obj.int_id_ = uuid.UUID(attributes['int_id'])
        return obj
        


    def add_molecular_imaging_data(self,
                                   mi: BaseMolecularImaging,
                                   register_to_primary_image=True,
                                   reference_image: numpy.array = None):
        """
        Adds molecular imaging data to section.
        
        If register_to_primary_image is True, registers the molecular imaging data to the primary image.
        reference_image: Needs to be set it molecular imaging data is to be registered to the primary image.
        """
        # TODO: Add this function to loading from config!
        if register_to_primary_image:
            warped_mi = register_to_ref_image(self.image.data, reference_image, mi)
        else:
            warped_mi = mi
        self.molecular_datas.append(warped_mi)

    def flip(self, axis: int = 0):
        """
        Flip all spatial data in this section.
        """
        self.image.flip(axis=axis)
        if self.segmentation_mask is not None:
            self.segmentation_mask.flip(axis=axis)
        if self.landmarks is not None:
            self.landmarks.flip(self.image.data.shape, axis=axis)
        if self.molecular_data is not None:
            self.molecular_data.flip(axis=axis)
        for mol_data in enumerate(self.molecular_datas):
            mol_data.flip(axis=axis)
        for annotation in self.annotations:
            annotation.flip(axis=axis)

    @classmethod
    def from_config(cls, config: Dict[str, str]):
        name = config['name']
        image_path = config['image_path']
        image = Image(data=io.imread(image_path))
        id_ = int(config['id'])
        segmenation_mask = None
        if 'segmentation_mask_path' in config:
            segmenation_mask_path = config['segmentation_mask_path']
            segmenation_mask = Annotation(data=io.imread(segmenation_mask_path))
        landmarks = None
        landmarks_path = config.get('landmarks_path', None)
        if landmarks_path is not None:
            landmarks = Pointset(data=pd.read_csv(landmarks_path))
        molecular_data = None
        if 'molecular_imaging_data' in config:
            molecular_data = load_molecular_imaging_data(config['molecular_imaging_data'])
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

        return cls(image=image,
                   name=name,
                   segmentation_mask=segmenation_mask,
                   landmarks=landmarks,
                   molecular_data=molecular_data,
                   id_=id_,
                   annotations=annotations,
                   config=config)

    @classmethod
    def from_directory(cls, directory):
        if not exists(directory):
            # Throw error
            pass
        with open(join(directory, 'config.json')) as f:
            config = json.load(f)
        name = config['name']
        id_ = config['id']
        image_path = join(directory, 'image.nii.gz')
        image = Image(sitk.GetArrayFromImage(sitk.ReadImage(image_path)))
        segmentation_mask_path = join(directory, 'mask.nii.gz')
        segmentation_mask = None
        if exists(segmentation_mask_path):
            segmentation_mask = Annotation(sitk.GetArrayFromImage(sitk.ReadImage(segmentation_mask_path)))
        landmarks = None
        landmarks_path = join(directory, 'landmarks.csv')
        if exists(landmarks_path):
            landmarks = Pointset(pd.read_csv(landmarks_path))
        molecular_data = None
        if 'molecular_imaging_data' in config:
            mol_data_path = join(directory, 'moldata')
            data_type = config['molecular_imaging_data']['data_type']
            molecular_data = load_molecular_imaging_data_from_directory(data_type, mol_data_path)
        annotation_dirs = [x for x in os.listdir(directory) if x.startswith('annotation_')]
        annotation_dirs.sort(key=lambda x: int(x.split('_')[-1]))
        annotations = []
        for annotation_dir in annotation_dirs:
            path = glob.glob(join(directory, annotation_dir, '*.nii.gz'))[0]
            label_path = join(directory, annotation_dir, 'labels.txt')
            ann_img = sitk.GetArrayFromImage(sitk.ReadImage(path))
            if exists(label_path):
                with open(label_path) as f:
                    labels = [x.strip() for x in f.readlines()]
                annotation = Annotation(ann_img, labels)
            else:
                annotation = Annotation(ann_img)
            # annotation = Annotation(sitk.GetArrayFromImage(sitk.ReadImage(path)))
            annotations.append(annotation)

        return cls(
            image=image,
            name=name,
            segmentation_mask=segmentation_mask,
            landmarks=landmarks,
            molecular_data=molecular_data,
            id_=id_,
            annotations=annotations,
            config=config
        )
        
    def init_mol_data(self):
        """
        Loads molecular data, in case any operation (like establishing file streams etc.) should 
        only be executed ones for performance reasons.
        """
        pass
