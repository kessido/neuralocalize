import localize
import utils.utils

class Args():
    def __init__(self, input_dir, task_filename, task_ordered_subjects_filename):
        self.input_dir = input_dir
        self.task_filename = task_filename
        self.task_ordered_subjects_filename = task_ordered_subjects_filename

subjects = [utils.utils.Subject(
        '100307',
        sessions_nii_paths=[
        r'..\test_resources\rfMRI_REST1_LR\rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST1_RL\rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST2_LR\rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii',
        r'..\test_resources\rfMRI_REST2_RL\rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii'
    ])]

args = Args('../test_resources/','AllSubjects_001.dtseries.nii' ,'subjects.txt')
