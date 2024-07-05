========================
Spatial-Omics Data Types
========================


Spatial omics data (SO-data) types are data types that require more preprocessing in order to make the data integratable in accordance with MIIT's workflow. For each
SO-data, we implement methods that map from the original SO-data type into a `reference_matrix` file format, in which each data point is presented in a coordinate matrix.

`reference_matrix` objects are at the core of MIIT's integration and are used for computing the overlap between different SO data types.

Each SO-data will extend the `BaseSpatialOmics` class.

We shortly highlight the most important attributes:

- `image`: Aligned image presentation of molecular imaging data. Typically an image of the stained histology.

- `reference_matrix`: Grid projection of SO-data into a grid system. Each value in the `reference_matrix` is either a `reference` or a background pixel, 
which can be set in `background`.

- `spec_to_ref_map`: Mapping of each SO-datapoint to a unique integer identifier. E.g. in Visium each SO-datapoint is identified by a unique barcode. Used to compute
overlap of `reference_matrices`.


In addition, so-data implements several basic image transformation operations.

-----------------------
Extending SO-data types
-----------------------


New SO-data types can be added as well. For examples, see `visium.py` and `imzml.py`. In order to add support for storing new SO-data types that are embedded in 
`Section` objects, new SO-data types can be added to the `SpatialDataLoader` class (see example).






--------------------
Spot scaling journal
--------------------