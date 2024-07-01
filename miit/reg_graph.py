"""
The Integrator class is essentially a graph model with each of the 
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Tuple 

from miit.spatial_data.section import Section
from miit.registerers.base_registerer import Registerer
from miit.registerers.registrations import Registration
from miit.utils.utils import clean_configs, filter_node_ids

@dataclass
class RegGraph:
    
    sections: Dict[int, Section]
    default_registerer: Registerer
    registrations: Dict[Section, Registration]
    name: str = ''
    registerers: Optional[Dict[Tuple[int, int], Registerer]] = None

    def register_sections(self, 
                          moving_section: Section, 
                          fixed_section: Section, 
                          args: Optional[Dict[Any, Any]] = None) -> Registration:
        if args is None:
            args = {}
        # Per default use masks.
        transformation = self.default_registerer.register_images(moving_img=moving_section.reference_image, 
                                                                   moving_img_mask=moving_section.segmentation_mask,
                                                                   fixed_img=fixed_section.reference_image, 
                                                                   fixed_img_mask=fixed_section.segmentation_mask,
                                                                   args=args)
        warped_section = Section.apply_transform(moving_section, self.default_registerer, transformation, args=args)
        registration = Registration(transformation_model=transformation,
                                    moving_section_id=moving_section._id,
                                    fixed_section_id=fixed_section._id,
                                    warped_section=warped_section,
                                    application_modes=dict())
        return registration

    def register_sections_by_idx(self, moving_id: int, fixed_id: int) -> Registration:
        moving_section = self.sections[moving_id]
        fixed_section = self.sections[fixed_id]
        transformation = self.default_registerer.register_images(moving_img=moving_section.reference_image, fixed_img=fixed_section.reference_image)
        warped_section = Section.apply_transform(moving_section, self.default_registerer, transformation)
        registration = Registration(transformation_model=transformation,
                                    moving_section_id=moving_section._id,
                                    fixed_section_id=fixed_section._id,
                                    warped_section=warped_section,
                                    application_modes=dict())
        return registration

    def compute_registration_path(self, id_sequence: List[int], args: Optional[Dict[Any, Any]] = None) -> List[Registration]:
        if args is None:
            args = {}
        moving_section = self.sections[id_sequence[0]]
        registration_path = []
        for id_ in id_sequence[1:]:
            fixed_section = self.sections[id_]
            warped_registration = self.register_sections(moving_section=moving_section, fixed_section=fixed_section, args=args)
            registration_path.append(warped_registration)
            moving_section = warped_registration.warped_section
        return registration_path

    def get_section_by_list_idx(self, idx):
        values = list(self.sections.values())
        return values[idx]

    def rescale_sections(self, width, height):
        for key in self.sections:
            self.sections[key].resize(width=width, height=height)

    @classmethod
    def from_config(cls, config):
        name = config['name']
        sections_config = config['sections']
        sections = OrderedDict()
        for section_config in sections_config:
            section = Section.from_config(section_config)
            sections[section._id] = section
        default_registerer = None
        registerers = []
        registrations = {}
        return cls(name=name, sections=sections, default_registerer=default_registerer, registerers=registerers, registrations=registrations)

    @classmethod
    def from_config_path(cls, config_path: str, remove_additional_data: bool = False, filter_ids: Optional[List[int]] = None):
        with open(config_path, 'r') as f:
            config = json.load(f)
            if remove_additional_data:
                config = clean_configs(config)
            if filter_ids is not None:
                config = filter_node_ids(config, filter_ids)
        return RegGraph.from_config(config)
