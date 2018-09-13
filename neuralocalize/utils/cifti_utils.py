import itertools
import warnings

import cifti
import nibabel as nib
import numpy as np

from . import constants, utils



class BrainMap:
    """Represent a brain map object
    """

    def __init__(self, brain_map_object):
        """
        Initialize from a brain map object that was loaded from the nii files. 
        """
        self.brain_structure_name = brain_map_object.brain_structure
        self.data_indices = range(
            brain_map_object.index_offset,
            brain_map_object.index_offset + brain_map_object.index_count)
        if brain_map_object.vertex_indices is not None:
            self.surface_indices = brain_map_object.vertex_indices._indices
        else:
            self.surface_indices = []


def load_cifti_brain_data_from_file(nii_path):
    """
    Convert dtseries.nii object into numpy matrix, and brain model meta data
    :param nii_path: A path to a nii file
    :return: numpy matrix containing the image data, brain models iterable
    """
    print("Loading cifti file:", nii_path)
    nib_data = nib.load(nii_path)
    cifti_image = np.array(np.array(nib_data.dataobj), dtype=constants.DTYPE)
    brain_maps = [BrainMap(i) for i in nib_data.header.matrix.get_index_map(1).brain_models]
    return cifti_image, brain_maps


def get_cortex_and_sub_cortex_indices(brain_maps):
    """Generate two list: The cortex indices and the sub-cortex indices.

    :param sample_file_path: the file path to load from the cortex indices.
    :return: (cortex indices list, sub-cortex indices list)
    """
    ctx_inds = list(itertools.chain(
        brain_maps[0].data_indices, brain_maps[1].data_indices))
    sub_ctx_inds = utils.remove_elements_from_list(range(91282), ctx_inds)
    return np.array(ctx_inds), np.array(sub_ctx_inds)


def save_cifti(cifti_img, path, series=None, brain_maps=None, sample_file_path='../resources/example.dtseries.nii'):
    """Save the cifti image to path.

    :param cifti_img: [n_component, 91282] The cifti image to save.
    :param path: The file path to save the cifti image to.
    :param series: The time series to use when saving the file.
                If not provided, create default one which fit the size of the image.
    :param brain_maps: If provided add this as the BrainMap inside the cifti file.
                If not take default one from sample file name.
    :param sample_file_path: An example cifti file
                that is used to generate the default brain maps if needed.
    """
    if not path.endswith('.dtseries.nii'):
        warnings.warn('Cifti files should be saved with ".dtseries.nii" file extension')

    # todo(kessi, itay) need to fit the time series to the actual time series.
    if series is not None and series.size != cifti_img.shape[0]:
        warnings.warn(
            ("The series provided in save_cifti under utils.cifti_utils " +
             "does not match the cifti image size provided: " +
             "cifti image shape: {}, series size: {}").format(cifti_img.shape, series.size)
        )
        series = None

    if series is None:
        series = cifti.Series(start=0.0, step=0.72, size=cifti_img.shape[0])

    if brain_maps is None:
        _, axis = cifti.read(sample_file_path)
        brain_maps = axis[1]

    cifti.write(path, cifti_img, (series, brain_maps))
