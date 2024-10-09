import numpy, numpy as np
import pandas

from miit.spatial_data.spatial_omics.imaging_data import BaseSpatialOmics


def map_counts_to_ref_mat(counts: pandas.core.frame.DataFrame, bmi: BaseSpatialOmics) -> numpy.ndarray:
    spec_to_ref_map = bmi.get_spec_to_ref_map()
    ref_mat = bmi.ref_mat.data.astype(int)
    count_map = {}
    for idx, row in counts.iterrows():
        rm_key = spec_to_ref_map[idx]
        count_map[rm_key] = row.iloc[0]
    indexer = np.array([count_map.get(i, 0) for i in range(ref_mat.min(), ref_mat.max() + 1)])
    measurement_mat = indexer[(ref_mat - ref_mat.min())]    
    return measurement_mat