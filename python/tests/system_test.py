import numpy as np
import traceback

import feature_extraction as feature_extraction
from utils.cifti_utils import load_cifti_brain_data_from_file
from utils.utils import Subject

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
    image, _ = load_cifti_brain_data_from_file(nii_path)
    # TODO this might be more complicated: may load separately for each cifti.
    # for the ICA matrix we needed:
    # np.array(nib_data.dataobj)
    # This might be different for every nii file.
    # TODO(loya) make sure np.array and not np.matrix, potential bug.
    return image


# TODO(loya) when we have a list of methods and files, create a decorator and run.
# TODO when done, change into 5 different tests and one end to end.
def abstract_test(method_to_test, nii_path):
    """Test pattern for the module: runs the method and checks that the result is the same.
    :param method_to_test: The method we wish to test
    :param nii_path: The path holding the expected matlab matrix.
    """
    actual_output = method_to_test()


# todo(kess) Ask Noam how to integrate with her tests.
def run_get_subcortical_parcellation_test():
    cifti_image, brain_models = load_cifti_brain_data_from_file(
        r'..\test_resources\GROUP_PCA_rand200_RFMRI.dtseries.nii')
    abstract_test(
        # TODO this path is not in the git. Should be added into resources
        lambda: feature_extraction.get_subcortical_parcellation(cifti_image, brain_models),
        r'..\..\matlab_results\SC_clusters.dtseries.nii')


def run_group_ica_separately_test():
    cifti_image, brain_models = load_cifti_brain_data_from_file(
        r'..\test_resources\GROUP_PCA_rand200_RFMRI.dtseries.nii')
    abstract_test(
        lambda: feature_extraction.run_group_ica_separately(cifti_image, brain_models)
        , r'..\..\matlab_results\ica_LR_MATCHED.dtseries.nii')


def run_group_ica_together_test():
    cifti_image, brain_models = load_cifti_brain_data_from_file(
        r'..\test_resources\GROUP_PCA_rand200_RFMRI.dtseries.nii')
    abstract_test(
        lambda: feature_extraction.run_group_ica_together(cifti_image, brain_models)
        , r'..\..\matlab_results\ica_both_lowdim.dtseries.nii')


def get_semi_dense_connectome_test():
    sc_cifti_image, _ = load_cifti_brain_data_from_file(
        r'..\test_resources\SC_clusters.dtseries.nii')

    # sc_cifti_image = np.ones([2, 4])
    # TODO(loya) validate these are the actual files.
    subjects = [Subject('noam',
                        left_right_hemisphere_data_path=r'..\test_resources\100307_DR2_nosmoothing.dtseries.nii',
                        sessions_nii_paths=[
                            r'..\test_resources\rfMRI_REST1_LR\rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii',
                            r'..\test_resources\rfMRI_REST1_RL\rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii',
                            r'..\test_resources\rfMRI_REST2_LR\rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii',
                            r'..\test_resources\rfMRI_REST2_RL\rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii'])]

    # subjects[0].left_right_hemisphere_data = np.ones([4, 4])
    # for session in subjects[0].sessions:
    #     as_line = np.arange(6*4) / 1000
    #     session.cifti = np.reshape(as_line, [6, 4])
    #     print("session.cifti:", session.cifti)
    feature_extraction.get_semi_dense_connectome(sc_cifti_image.transpose(), subjects)
    ret = subjects[0].correlation_coefficient

    actual_output, _ = load_cifti_brain_data_from_file(r'..\test_resources\noam_results\100307_RFMRI_nosmoothing.dtseries.nii')
    print("Diff norm:", np.linalg.norm(ret - actual_output))
    print("RET shape:", ret.shape, "actual shape:", actual_output.shape)
    print("RESULT:", np.allclose(actual_output, ret))


def run_dual_regression_test():
    dt, brain_models = load_cifti_brain_data_from_file(
        r'..\test_resources\ica_LR_MATCHED.dtseries.nii')

    # TODO(loya) notice there are more parameters such as ROIs
    subjects = [Subject(
        'noam',
        sessions_nii_paths=[
            r'..\test_resources\Subjects\100307\MNINonLinear\Results\rfMRI_REST1_LR\rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii',
            r'..\test_resources\Subjects\100307\MNINonLinear\Results\rfMRI_REST1_RL\rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii',
            r'..\test_resources\Subjects\100307\MNINonLinear\Results\rfMRI_REST2_LR\rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii',
            r'..\test_resources\Subjects\100307\MNINonLinear\Results\rfMRI_REST2_RL\rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii'
        ])]

    # TODO(loya) this is out of date. The test doesn't return value, but updates the .left_right... field in subjects.
    feature_extraction.run_dual_regression(dt.transpose(), brain_models, subjects)
    print(subjects[0].left_right_hemisphere_data)
    # abstract_test(
    #     lambda: feature_extraction.run_dual_regression(dt.transpose(), brain_models, subjects)
    #     , r'..\test_resources\100307_DR2_nosmoothing.dtseries.nii')


# try:
#     run_group_ica_together_test()
# except Exception:
#     traceback.print_exc()
# try:
#     run_group_ica_separately_test()
# except Exception:
#     traceback.print_exc()
# try:
#     get_semi_dense_connectome_test()
# except Exception:
#     traceback.print_exc()
# try:
#     run_get_subcortical_parcellation_test()
# except Exception:
#     traceback.print_exc()
# run_dual_regression_test()
run_dual_regression_test()