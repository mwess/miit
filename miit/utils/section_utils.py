from typing import Any, Dict, List, Tuple

from miit.spatial_data import Section
from .image_utils import get_padding_params


def get_symmetric_padding_for_sections(sections: List[Section]) -> Dict[Any, Tuple[int, int, int, int]]:
    paddings = {}
    max_size = get_max_size_from_sections(sections)
    for section in sections:
        paddings[section._id] = get_padding_params(section.reference_image.data, max_size)
    return paddings


def get_section_max_size(section: Section) -> int:
    return max(section.reference_image.data.shape[0], section.reference_image.data.shape[1])


def get_max_size_from_sections(sections: List[Section]) -> int:
    max_size = -1
    for section in sections:
        max_size = max(max_size, get_section_max_size(section))
    return max_size
