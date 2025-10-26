import matplotlib.pyplot as plt
import numpy as np

from miit.spatial_data.base_types import BaseImage
from miit.spatial_data.section import Section

# TODO: Remove
def clean_configs(config: dict) -> dict:
    for section in config['sections']:
        if 'molecular_imaging_data' in section:
            del section['molecular_imaging_data']
    return config


def filter_node_ids(config: dict, id_list: list) -> dict:
    keep_sections = []
    for section in config['sections']:
        # print(section['id'])
        if section['id'] in id_list:
            keep_sections.append(section)
    config['sections'] = keep_sections
    return config


# TODO: I think thise whole file can be removed.
def plot_registration_summary(moving: Section, fixed: Section, warped: Section, save_path: str, with_landmarks: bool=True):
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    img = moving.reference_image.data.astype(int)
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Moving')
    img = fixed.reference_image.data.astype(int)
    axs[1].imshow(img)
    axs[1].axis('off')
    axs[1].set_title('Fixed')
    img = warped.reference_image.data.astype(int)
    axs[2].imshow(img)
    axs[2].axis('off')
    axs[2].set_title('Warped')
    if with_landmarks:
        for idx, row in moving.landmarks.data.iterrows():
            x1 = row['x']
            y1 = row['y']
            axs[0].plot(x1, y1, 'b.')
        for idx, row in fixed.landmarks.data.iterrows():
            x1 = row['x']
            y1 = row['y']
            axs[1].plot(x1, y1, 'g.')
        warped_lms = warped.landmarks.data.set_index('label')
        fixed_lms = fixed.landmarks.data.set_index('label')
        joined_lms = warped_lms.join(fixed_lms, rsuffix='_fixed')
        for idx, row in joined_lms.iterrows():
            x1 = row['x']
            y1 = row['y']
            x2 = row['x_fixed']
            y2 = row['y_fixed']
            if any(map(lambda x: np.isnan(x) or np.isinf(x), [x1, y1, x2, y2])):
                continue
            axs[2].plot(x1, y1, 'b.')
            axs[2].plot(x2, y2, 'g.')
            x_values = [x1, x2]
            y_values = [y1, y2]
            axs[2].plot(x_values, y_values, 'k', linestyle="-")
    fig.savefig(save_path)
    fig.clf()
    plt.close()


def plot_with_landmarks(section: Section):
    plt.imshow(section.reference_image.data.astype(int))
    plt.title('Image with landmarks')
    plt.axis('off')
    for idx, row in section.landmarks.data.iterrows():
        x1 = row['x']
        y1 = row['y']
        plt.plot(x1, y1, 'b.')


def plot_sections_with_landmark_distance(image: BaseImage, unified_lms):
    plt.imshow(image.data/255.)
    plt.title('Warped Image')
    plt.axis('off')
    for idx, row in unified_lms.iterrows():
        x1 = row['x_src']
        y1 = row['y_src']
        x2 = row['x_dst']
        y2 = row['y_dst']
        plt.plot(x1, y1, 'b.')
        plt.plot(x2, y2, 'g.')
        x_values = [x1, x2]
        y_values = [y1, y2]
        plt.plot(x_values, y_values, 'k', linestyle="-")



# TODO: Remove function.
def find_ided_peaks(peaks, peak_table, mass_error_mz=2.00000):
    peak_max_diffs = np.abs(peak_table['m/z'] - peak_table['m/z'] * (1 + mass_error_mz))
    _ret = []
    for _idxs, _mzs in peaks:
        _t_idx = []
        _t_mz = []
        _t_id = []
        for _idx, _mz in zip(_idxs, _mzs):
            peak_table_val = peak_table['m/z']
            diffs = np.abs(peak_table_val - _mz)
            sub_table = peak_table[diffs < peak_max_diffs]
            if sub_table.shape[0] > 0:
                _t_idx.append(_idx)
                _t_mz.append(_mz)
                _t_id.append(peak_table['ID'].iloc[(np.abs(peak_table['m/z'] - _mz)).argmin()])
        _ret.append((_t_idx, _t_mz, _t_id))
    return _ret


import matplotlib.pyplot as plt
import numpy as np

from miit.spatial_data import BaseImage
from miit.spatial_data.section import Section


# TODO: I think thise whole file can be removed.
def plot_registration_summary(moving: Section, fixed: Section, warped: Section, save_path: str, with_landmarks: bool=True):
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    img = moving.reference_image.data.astype(int)
    axs[0].imshow(img)
    axs[0].axis('off')
    axs[0].set_title('Moving')
    img = fixed.reference_image.data.astype(int)
    axs[1].imshow(img)
    axs[1].axis('off')
    axs[1].set_title('Fixed')
    img = warped.reference_image.data.astype(int)
    axs[2].imshow(img)
    axs[2].axis('off')
    axs[2].set_title('Warped')
    if with_landmarks:
        for idx, row in moving.landmarks.data.iterrows():
            x1 = row['x']
            y1 = row['y']
            axs[0].plot(x1, y1, 'b.')
        for idx, row in fixed.landmarks.data.iterrows():
            x1 = row['x']
            y1 = row['y']
            axs[1].plot(x1, y1, 'g.')
        warped_lms = warped.landmarks.data.set_index('label')
        fixed_lms = fixed.landmarks.data.set_index('label')
        joined_lms = warped_lms.join(fixed_lms, rsuffix='_fixed')
        for idx, row in joined_lms.iterrows():
            x1 = row['x']
            y1 = row['y']
            x2 = row['x_fixed']
            y2 = row['y_fixed']
            if any(map(lambda x: np.isnan(x) or np.isinf(x), [x1, y1, x2, y2])):
                continue
            axs[2].plot(x1, y1, 'b.')
            axs[2].plot(x2, y2, 'g.')
            x_values = [x1, x2]
            y_values = [y1, y2]
            axs[2].plot(x_values, y_values, 'k', linestyle="-")
    fig.savefig(save_path)
    fig.clf()
    plt.close()


def plot_with_landmarks(section: Section):
    plt.imshow(section.reference_image.data.astype(int))
    plt.title('Image with landmarks')
    plt.axis('off')
    for idx, row in section.landmarks.data.iterrows():
        x1 = row['x']
        y1 = row['y']
        plt.plot(x1, y1, 'b.')

def plot_sections_with_landmark_distance(image: BaseImage, unified_lms):
    plt.imshow(image.data/255.)
    plt.title('Warped Image')
    plt.axis('off')
    for idx, row in unified_lms.iterrows():
        x1 = row['x_src']
        y1 = row['y_src']
        x2 = row['x_dst']
        y2 = row['y_dst']
        plt.plot(x1, y1, 'b.')
        plt.plot(x2, y2, 'g.')
        x_values = [x1, x2]
        y_values = [y1, y2]
        plt.plot(x_values, y_values, 'k', linestyle="-")


# TODO: Can this function be removed. Duplicate in imzml.py?
def map_accumulated_data_to_imzml(target_bmi: Imzml,
                                  accumulated_df: pandas.DataFrame,
                                  output_path: str,
                                  mzs: numpy.ndarray | None = None):
    spec_to_ref_map = target_bmi.get_spec_to_ref_map()
    template_imzml = ImzMLParser(target_bmi.config['imzml'])
    if mzs is None:
        mzs = np.array([float(x) for x in accumulated_df.columns])
    imzml_writer = ImzMLWriter(output_path)
    spec_to_ref_map = target_bmi.get_spec_to_ref_map()
    for msi_idx, coords in enumerate(template_imzml.coordinates):
        acc_df_idx = spec_to_ref_map[msi_idx]
        intensities = accumulated_df.loc[acc_df_idx].to_numpy()
        imzml_writer.addSpectrum(mzs, intensities, coords)
    imzml_writer.finish()