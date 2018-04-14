import pytest
import nibabel as nib
import numpy as np

import src.feature_extraction as feature_extraction

method_to_nii = [
    (feature_extraction.run_group_ica_separately, 'nii_path'),
    (feature_extraction.run_group_ica_together, 'nii_path'),
    (feature_extraction.run_dual_regression, 'nii_path'),
    (feature_extraction.get_subcortical_parcellation, 'nii_path'),
    (feature_extraction.get_semi_dense_connectome, 'nii_path'),
]


def get_matlab_matrix_as_numpy(nii_path):
    """Converts nii object into numpy matrix.
    :param nii_path: A path to the nii file.
    :return: numpy matrix with the same data.
    """
    nib_data = nib.load(nii_path)  # TODO this might be more complicated: may load separately for each cifti.
    # for the ICA matrix we needed:
    # np.array(nib_data.dataobj)
    # This might be different for every nii file.
    return np.array(nib_data.dataobj)  # TODO(loya) make sure np.array and not np.matrix, potential bug.


# TODO(loya) when we have a list of methods and files, create a decorator and run.
# TODO when done, change into 5 different tests and one end to end.
def abstract_test(method_to_test, nii_path):
    """Test pattern for the module: runs the method and checks that the result is the same.
    :param method_to_test: The method we wish to test
    :param nii_path: The path holding the expected matlab matrix.
    """
    actual_output = method_to_test()  # TODO(loya) handle params if needed.
    expected_output = get_matlab_matrix_as_numpy(nii_path)
    assert np.allclose(actual_output, expected_output)


def get_matlab_matrix_as_numpy_brain_subcortical_data(nii_path):
    """
    Convert nii object into matlab matrix, and brain model meta data
    :param nii_path: A path to a nii file
    :return: numpy matrix containing the image data, brain models iterable
    """
    nib_data = nib.load(nii_path)
    return np.array(nib_data.dataobj), nib_data.header.matrix.get_index_map(1).brain_models


# todo(kess) Ask Noam how to integrate with her tests.
def run_get_subcortical_parcellation_test():
    cifti_image, brain_models = get_matlab_matrix_as_numpy_brain_subcortical_data(
        'GROUP_PCA_rand200_RFMRI.dtseries.nii')
    abstract_test(
        lambda: feature_extraction.get_subcortical_parcellation(cifti_image, brain_models)
        , r'..\..\matlab_results\SC_clusters.dtseries.nii')


run_get_subcortical_parcellation_test()
