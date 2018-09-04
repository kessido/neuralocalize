import numpy as np
import nibabel as nib
import utils.utils as util
import utils.cifti_utils
import pytest
import prediction
import localize


class Args():
    def __init__(self, input_dir, task_filename, task_ordered_subjects_filename):
        self.input_dir = input_dir
        self.task_filename = task_filename
        self.task_ordered_subjects_filename = task_ordered_subjects_filename


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
        '100307',
        sessions_nii_paths=[
            r'..\test_resources\rfMRI_REST1_LR\rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii',
            r'..\test_resources\rfMRI_REST1_RL\rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii',
            r'..\test_resources\rfMRI_REST2_LR\rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii',
            r'..\test_resources\rfMRI_REST2_RL\rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii'
        ])]

    args = Args('../test_resources/', 'AllSubjects_001.dtseries.nii', 'subjects.txt')
    tasks = localize.load_subjects_task(args, subjects)
    # model = prediction.Localizer(subjects, subjects_task=tasks, pca_result=pca_result)
    # model.save_to_file('model.pcl.gz')
    model = localize.Localizer.load_from_file('model.pcl.gz')
    res = model.predict(subjects, load_feature_extraction=True,
                feature_extraction_path='feature_ext_result.mat')
    utils.cifti_utils.save_cifti(res, 'res.dtseries.nii')


dummy_test_localizer_run()
# dummy_test_feature_extraction_run()
