from miit.spatial_data import Annotation

from .utils import (
    make_circle_image_multichannel, 
    make_circle_image_singlechannel,
    get_circle_image_labels
)


def test_annotation_channel_mode_default():
    np_img = make_circle_image_multichannel(draw_c2=False)
    ann = Annotation(data=np_img)
    assert not ann.is_multichannel and len(ann.labels) == 1


def test_annotation_default_labels():
    np_img = make_circle_image_multichannel()
    ann = Annotation(data=np_img, is_multichannel=True)
    assert len(ann.labels) == 2


def test_annotation_multichannel_labels():
    np_img = make_circle_image_multichannel()
    labels = get_circle_image_labels()
    labels = {labels[i]: i + 1 for i in range(len(labels))}
    ann = Annotation(data=np_img, labels=labels, is_multichannel=True)
    assert set(labels) == set(ann.labels)


def test_annotation_singlechannel_labels():
    np_img = make_circle_image_singlechannel()
    labels = get_circle_image_labels()
    ann = Annotation(data=np_img, labels=labels)
    assert ann.data.shape[-1] == len(ann.labels)


def test_convert_singlechannel_to_multichanel():
    np_img = make_circle_image_singlechannel()
    labels = get_circle_image_labels()
    ann = Annotation(data=np_img, labels=labels)
    ann.convert_to_multichannel()
    assert len(ann.data.shape) == 2 and isinstance(ann.labels, dict)


def test_convert_multichannel_to_singlechannel():
    np_img = make_circle_image_multichannel()
    labels = get_circle_image_labels()
    labels = {labels[i]: i + 1 for i in range(len(labels))}
    ann = Annotation(data=np_img, labels=labels, is_multichannel=True)
    ann.convert_to_singlechannel()
    assert len(ann.data.shape) == 3 and ann.data.shape[2] == 2 and isinstance(ann.labels, list)