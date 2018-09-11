import traceback

import numpy as np

import constants
import localize
import prediction
import utils.cifti_utils
import utils.utils as util


class Args():
    def __init__(self, input_dir, task_filename, task_ordered_subjects_filename):
        self.input_dir = input_dir
        self.task_filename = task_filename
        self.task_ordered_subjects_filename = task_ordered_subjects_filename


def dummy_test_feature_extraction_run():
    pca_result, _ = utils.cifti_utils.load_cifti_brain_data_from_file(
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
    pca_result, _ = utils.cifti_utils.load_cifti_brain_data_from_file(
        r'..\test_resources\GROUP_PCA_rand200_RFMRI.dtseries.nii')
    subjects = [util.Subject(
        '100307').load_from_directory(f'..\\test_resources\\Subjects\\100307\\{constants.PATH_TO_SESSIONS}')]

    args = Args(r'../test_resources/', 'AllSubjects_001.dtseries.nii', 'subjects.txt')
    tasks = localize.load_subjects_task(args, subjects)
    model = prediction.Localizer(subjects, pca_result=pca_result)
    model.fit(subjects, tasks)
    model.save_to_file('model.pcl.gz')
    model = localize.Localizer.load_from_file('model.pcl.gz')
    res = model.predict(subjects)
    utils.cifti_utils.save_cifti(res, 'res.dtseries.nii')
    print('Norm:', np.linalg.norm(tasks[0] - res[0]))


try:
    dummy_test_localizer_run()
except:
    traceback.print_exc()

try:
    pass  # dummy_test_feature_extraction_run()
except:
    traceback.print_exc()
