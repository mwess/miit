================
Basic Data Types
================


MIIT provides wrapper classes to ease handling of various different data format. By design, underlying data in each type of image
can be accessed through the `data` attribute.



------------
DefaultImage
------------

This is the wrapper class for handling all default image data. Internally images are processed as 3 channel numpy arrays in shape of W x H x C. During image transformation
a 'LINEAR' interpolation approach is used. 


----------
Annotation
----------

This class is similar to `DefaultImage`. The main difference is that `Annotation` data can be of shape W x H or W x H x C where C presents different classes. 
Additionally, labels can be provided to name each annotation class. 'Nearest Neighbor' interpolation is used during image transformation.

If labels are provided, a utility method can be used to get an annotation mask for a given label:

.. code-block:: python

    annotation = Annotation(data=..., labels=...)
    mask = annotation.get_by_label('label')


--------
Pointset
--------

This class can be used to handle coordinate valued data such as landmarks. Internally, landmarks are handled as pandas Dataframes. 

------------
GeoJSONData
------------

Wrapper class for geojson data. 

Includes a utility class for converting `GeoJSONData` to `Annotation`:

.. code-block:: python

    geojson_data = GeoJSONData(...)
    converted_annotation = geojson_data.to_annotation(reference_image=reference_image)


By default, feature ids are used as labels for `Annotation`. Optionally, a `label_function` can be supplied to control how `labels` in `annotations` are 
extracted from each feature. E.g.:

.. code-block:: python

    def get_labels_from_feature(feature):
        if 'classification' in feature['properties']:
            return feature['properties']['classification']['name']
        elif 'name' in feature['properties']:
            return feature['properties']['name']
        else:
            return feature['id']

    geojson_data = GeoJSONData(...)
    converted_annotation = geojson_data.to_annotation(reference_image=reference_image)
