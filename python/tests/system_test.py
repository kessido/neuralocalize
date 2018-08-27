import pytest
import nibabel as nib
import numpy as np

from utils.cifti_utils import load_nii_brain_data_from_file, BrainMap
import feature_extraction as feature_extraction

method_to_nii = [
    (feature_extraction.run_group_ica_separately, 'nii_path'),
    (feature_extraction.run_group_ica_together, 'nii_path'),
    (feature_extraction.run_dual_regression, 'nii_path'),
    (feature_extraction.get_subcortical_parcellation, 'nii_path'),
    (feature_extraction.get_semi_dense_connectome, 'nii_path'),
]

class Session(object):
    """A class representing a session"""
    def __init__(self, cifti):
        self.cifti = cifti

class Subject(object):
    """A class containing a subject and everything related to it"""

    def __init__(self, sessions):
        self.sessions = sessions

def get_matlab_matrix_as_numpy(nii_path):
    """Converts nii object into numpy matrix.
    :param nii_path: A path to the nii file.
    :return: numpy matrix with the same data.
    """
    nib_data = nib.load(
        nii_path)  # TODO this might be more complicated: may load separately for each cifti.
    # for the ICA matrix we needed:
    # np.array(nib_data.dataobj)
    # This might be different for every nii file.
    # TODO(loya) make sure np.array and not np.matrix, potential bug.
    return np.array(nib_data.dataobj)


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

# todo(kess) Ask Noam how to integrate with her tests.
def run_get_subcortical_parcellation_test():
    cifti_image, brain_models = load_nii_brain_data_from_file(
        'GROUP_PCA_rand200_RFMRI.dtseries.nii')
    abstract_test(
        # TODO this path is not in the git. Should be added into resources
        lambda: feature_extraction.get_subcortical_parcellation(cifti_image, brain_models), r'..\..\matlab_results\SC_clusters.dtseries.nii')

def run_group_ica_separately_test():
    cifti_image, brain_models = load_nii_brain_data_from_file(
        'GROUP_PCA_rand200_RFMRI.dtseries.nii')
    abstract_test(
        lambda: feature_extraction.run_group_ica_separately(cifti_image, brain_models)
        , r'..\..\matlab_results\ica_LR_MATCHED.dtseries.nii')

def run_group_ica_together_test():
    cifti_image, brain_models = load_nii_brain_data_from_file(
        'GROUP_PCA_rand200_RFMRI.dtseries.nii')
    abstract_test(
        lambda: feature_extraction.run_group_ica_together(cifti_image, brain_models)
        , r'..\..\matlab_results\ica_both_lowdim.dtseries.nii')

def run_get_semi_dense_connectome_test():
    cifti_image, brain_models = load_nii_brain_data_from_file(
        r'..\test_resources\TODO')
    subjects = [Subject([Session(cifti_image, brain_models)])]

    abstract_test(
        lambda: feature_extraction.get_semi_dense_connectome(subjects)
        , r'..\test_resources\TODO')

def run_get_semi_dense_connectome_test():
    cifti_image, brain_models = load_nii_brain_data_from_file(
        r'..\test_resources\TODO')
    subjects = [Subject([Session(cifti_image, brain_models)])]

    abstract_test(
        lambda: feature_extraction.run_dual_regression(subjects)
        , r'..\test_resources\TODO')

run_get_subcortical_parcellation_test()
