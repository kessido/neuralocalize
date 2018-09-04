import numpy as np
import nibabel as nib
import utils.utils as util
import pytest
import prediction

def load_nii_brain_data_from_file(nii_path):
    """
    Convert nii object into matlab matrix, and brain model meta data
    :param nii_path: A path to a nii file
    :return: numpy matrix containing the image data, brain models iterable
    """
    nib_data = nib.load(nii_path)
    return np.array(nib_data.dataobj)

def dummy_test_feature_extraction_run():
    pca_result = load_nii_brain_data_from_file(
        '../test_resources/GROUP_PCA_rand200_RFMRI.dtseries.nii')
    subjects = [util.Subject(
        'noam',
        sessions_nii_paths=[
        r'..\test_resources\rfMRI_REST1_LR\rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST1_RL\rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST2_LR\rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST2_RL\rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii'
    ])]
    prediction.FeatureExtractor(subjects, pca_result, r'..\resources\example.dtseries.nii')

def dummy_test_localizer_run():
    pca_result = load_nii_brain_data_from_file(
        '../test_resources/GROUP_PCA_rand200_RFMRI.dtseries.nii')
    subjects = [util.Subject(
        'noam',
        sessions_nii_paths=[
        r'..\test_resources\rfMRI_REST1_LR\rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST1_RL\rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST2_LR\rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST2_RL\rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii'
    ])]

    prediction.Localizer(subjects, pca_result=pca_result)

dummy_test_localizer_run()
# dummy_test_feature_extraction_run()