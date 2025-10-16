import numpy, numpy as np
import pandas


def eucl(src: numpy.ndarray, dst: numpy.ndarray) -> float:
    return np.sqrt(np.square(src[:, 0] - dst[:, 0]) + np.square(src[:, 1] - dst[:, 1])) # type: ignore


def compute_distance_for_lm(warped_df: pandas.DataFrame, 
                            fixed_df: pandas.DataFrame) -> pandas.DataFrame:
    merged_df = warped_df.merge(fixed_df, on='label', suffixes=('_src', '_dst'))
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df.dropna(inplace=True)
    src_mat = merged_df[['x_src', 'y_src']].to_numpy()
    dst_mat = merged_df[['x_dst', 'y_dst']].to_numpy()
    merged_df['tre'] = eucl(src_mat, dst_mat)
    return merged_df


def compute_tre(target_lms: pandas.DataFrame, 
                warped_lms: pandas.DataFrame, 
                shape: tuple[int, int]) -> tuple[float, float, float, float]:
    """Compute target registration errors between two pointsets.

    Args:
        target_lms (pandas.DataFrame): _description_
        warped_lms (pandas.DataFrame): _description_
        shape (Tuple[int, int]): _description_

    Returns:
        Tuple[float, float, float, float]: Returns mean_rtre, median_rtre, mean_tre, median_tre
    """
    unified_lms = compute_distance_for_lm(warped_lms, target_lms)
    diag = np.sqrt(np.square(shape[0]) + np.square(shape[1]))
    unified_lms['rtre'] = unified_lms['tre']/diag
    mean_rtre = float(np.mean(unified_lms['rtre']))
    median_rtre = float(np.median(unified_lms['rtre']))
    median_tre = float(np.median(unified_lms['tre']))
    mean_tre = float(np.mean(unified_lms['tre']))
    return mean_rtre, median_rtre, mean_tre, median_tre