import numpy, numpy as np

def get_mappings(ref_mat1: numpy.array, 
                 ref_mat2: numpy.array, 
                 background1: float, 
                 background2: float) -> dict | set:
    mappings = compute_reference_matrix_mappings(ref_mat1, ref_mat2, background1)
    # Also get unique ids.
    unique_vals = set()
    for key in mappings:
        counts = mappings[key][0]
        for key2 in counts:
            if key2 != background2:
                unique_vals.add(key2)
    return mappings, unique_vals


def compute_reference_matrix_mappings(ref_mat1: numpy.array, 
                                      ref_mat2: numpy.array, 
                                      background1: float) -> Dict[int, Tuple[numpy.ndarray, numpy.ndarray]]:
    spots = {}
    for i in range(ref_mat1.shape[0]):
        for j in range(ref_mat1.shape[1]):
            val = ref_mat1[i,j]
            if val == background1:
                continue
            if val not in spots:
                spots[val] = []
            val2 = ref_mat2[i,j]
            spots[val].append(val2)
    for key in spots:
        spots[key] = np.unique(spots[key], return_counts=True)
    return spots

