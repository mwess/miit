from typing import Any

from miit.spatial_data import Section
from .image_utils import get_padding_params


def get_symmetric_padding_for_sections(sections: list[Section]) -> dict[Any, tuple[int, int, int, int]]:
    paddings = {}
    max_size = get_max_size_from_sections(sections)
    for section in sections:
        reference_image = section.reference_image
        if reference_image is None:
            raise Exception('No reference image found.')
        paddings[section._id] = get_padding_params(reference_image.size, max_size)
    return paddings


def get_section_max_size(section: Section) -> int | None:
    if section.reference_image is not None:
        return max(section.reference_image.size[0], section.reference_image.size[1])


def get_max_size_from_sections(sections: list[Section]) -> int:
    max_size = -1
    for section in sections:
        max_section_size = get_section_max_size(section)
        if max_section_size is None:
            raise Exception('No reference image size found for section.')
        max_size = max(max_size, max_section_size)
    return max_size
