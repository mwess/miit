from functools import partial
import pickle

import cv2
import numpy as np

from scipy import ndimage as ndi

from skimage import feature, future
from skimage.morphology import area_closing
from skimage.filters import gaussian
from skimage.measure import label, regionprops




features_func = partial(feature.multiscale_basic_features,
                        intensity=False, edges=True, texture=True,
                        sigma_min=1, sigma_max=16,
                        channel_axis=-1)


def preprocess_image_for_segmentation(img):
    img = gaussian(img, sigma=2, channel_axis=-1)
    img = cv2.resize(img, (512, 512))
    return img


def segment_histology_for_rfc_training(image, rfc, region_min_size=25000):
    img = preprocess_image_for_segmentation(image)
    features = features_func(img)
    prediction = future.predict_segmenter(features, rfc)
    prediction = prediction - 1
    prediction2 = area_closing(prediction, area_threshold=60, connectivity=2)
    filled_mask = ndi.binary_fill_holes(prediction2).astype(float)
    # prediction3 = area_closing(prediction2, area_threshold=60, connectivity=2)
    # filled_mask = ndi.binary_fill_holes(prediction3).astype(float)
    filled_mask = cv2.resize(filled_mask, (image.shape[1], image.shape[0])).astype(np.uint8)
    labels = label(filled_mask)
    new_mask = np.zeros(filled_mask.shape, dtype=np.uint8)
    for region in regionprops(labels):
        # print(region.area)
        if region.area >= region_min_size:
            minr, minc, maxr, maxc = region.bbox
            new_mask[minr:maxr, minc:maxc] = region.image.astype(np.uint8)
    return new_mask


def segment_image(image):
    with open('segmentation_models/rfc_all.pickle', 'rb') as f:
        rfc = pickle.load(f)
    predicted_mask = segment_histology_for_rfc_training(image, rfc, region_min_size=50000)
    return predicted_mask
