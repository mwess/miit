import numpy as np
import pandas

from miit.spatial_data.section import Section


def eucl(src, dst):
    return np.sqrt(np.square(src[:, 0] - dst[:, 0]) + np.square(src[:, 1] - dst[:, 1]))


def compute_distance_for_lm(warped_df: pandas.core.frame.DataFrame, fixed_df: pandas.core.frame.DataFrame):
    merged_df = warped_df.merge(fixed_df, on='label', suffixes=('_src', '_dst'))
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df.dropna(inplace=True)
    src_mat = merged_df[['x_src', 'y_src']].to_numpy()
    dst_mat = merged_df[['x_dst', 'y_dst']].to_numpy()
    merged_df['tre'] = eucl(src_mat, dst_mat)
    return merged_df


def compute_tre_sections_(target_section: Section, warped_section: Section):
    unified_lms = compute_distance_for_lm(warped_section.landmarks.data, target_section.landmarks.data)
    shape = target_section.image.data.shape
    image_diagonal = np.sqrt(np.square(shape[0]) + np.square(shape[1]))
    unified_lms['rtre'] = unified_lms['tre']/image_diagonal
    mean_rtre = np.mean(unified_lms['rtre'])
    median_rtre = np.median(unified_lms['rtre'])
    median_tre = np.median(unified_lms['tre'])
    mean_tre = np.mean(unified_lms['tre'])
    return mean_rtre, median_rtre, mean_tre, median_tre


def compute_tre(target_section, warped_section, shape):
    unified_lms = compute_distance_for_lm(warped_section.landmarks.data, target_section.landmarks.data)
    diag = np.sqrt(np.square(shape[0]) + np.square(shape[1]))
    unified_lms['rtre'] = unified_lms['tre']/diag
    mean_rtre = np.mean(unified_lms['rtre'])
    median_rtre = np.median(unified_lms['rtre'])
    median_tre = np.median(unified_lms['tre'])
    mean_tre = np.mean(unified_lms['tre'])
    return mean_rtre, median_rtre, mean_tre, median_tre