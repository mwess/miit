"""
Contains Registrations class which wraps around the registration data model
and provides additional meta information regarding the application of the registration model.

For now, a string should be enough to connect a registration model to its registerer and the registration mode.
"""
from dataclasses import dataclass
from typing import Any, Dict

from miit.spatial_data.section import Section

@dataclass
class Registration:
    
    transformation_model: Any
    moving_section_id: int
    fixed_section_id: int
    warped_section: Section
    application_modes: Dict[Any, Any]    
