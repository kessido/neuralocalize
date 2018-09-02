import nibabel as nib
import numpy as np
import itertools
import utils
from utils.utils import remove_elements_from_list


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


def load_nii_brain_data_from_file(nii_path):
    """
    Convert nii object into matlab matrix, and brain model meta data
    :param nii_path: A path to a nii file
    :return: numpy matrix containing the image data, brain models iterable
    """
    nib_data = nib.load(nii_path)
    return np.array(nib_data.dataobj), [BrainMap(i) for i in nib_data.header.matrix.get_index_map(1).brain_models]


def get_cortex_and_sub_cortex_indices(sample_file_path='./example.dtseries.nii'):
    _, brain_maps = load_nii_brain_data_from_file(sample_file_path)
    ctx_inds = itertools.chain(
        brain_maps[0].data_indices, brain_maps[1].data_indices)
    sub_ctx_inds = remove_elements_from_list(range(91282), ctx_inds)
    return ctx_inds, sub_ctx_inds
