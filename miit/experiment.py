
"""
The Integrator class is essentially a graph model with each of the 
"""
from collections import OrderedDict
from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from miit.spatial_data.section import Section
from miit.utils.utils import clean_configs, filter_node_ids

@dataclass
class Experiment:
    
    name: str = ''
    # Define all the data of the experiment
    sections: Dict[int, Section]
    
    def get_section_by_list_idx(self, idx):
        values = list(self.sections.values())
        return values[idx]

    def rescale_sections(self, width, height):
        for key in self.sections:
            self.sections[key].rescale_data(width=width, height=height)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Experiment':
        name = config['name']
        sections_config = config['sections']
        sections = OrderedDict()
        for section_config in sections_config:
            section = Section.from_config(section_config)
            sections[section.id_] = section
        return cls(name=name, sections=sections)

    @classmethod
    def from_config_path(cls, config_path: str, remove_additional_data: bool = False, filter_ids: Optional[List[int]] = None) -> 'Experiment':
        with open(config_path, 'r') as f:
            config = json.load(f)
            if remove_additional_data:
                config = clean_configs(config)
            if filter_ids is not None:
                config = filter_node_ids(config, filter_ids)
        return Experiment.from_config(config)
