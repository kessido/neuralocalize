import numpy as np

RFMRI_nosmoothing_filename = '_RFMRI_nosmoothing.dtseries.nii'
STRUCT_FILENAME = '_Struct.dtseries.nii'
FILTER_FILENAME = 'ica_both_lowdim.dtseries.nii'
MODEL_FILENAME = 'model.pcl.gzip'
DTYPE = np.float64
ICA_FUCKING_CONST = 0.005
SPATIAL_FILTERS_CONST = 0.005
SPATIAL_FILTERS_CONST_WITH_LOAD = 2.1
PATH_TO_SESSION = "MNINonLinear/Results"

SESSION_IDS = [('1', 'LR'), ('1', 'RL'), ('2', 'LR'), ('2', 'RL')]
PATH_TO_SESSIONS = "MNINonlinear/Results"
SESSION_NAME_TEMPLATE = "rfMRI_REST%s_%s/rfMRI_REST%s_%s_Atlas_MSMAll_hp2000_clean.dtseries.nii"
SESSION_DIRS = [SESSION_NAME_TEMPLATE % (num, side, num, side) for num, side in SESSION_IDS]
DEFAULT_TASK_FILENAME = ''
DEFAULT_TASK_ORDERED_SUBJ_FILE = ''

DEFAULT_NUMBER_OF_TASK = 86

DEFAULT_PCA_RESULT_PATH = r'../test_resources/GROUP_PCA_rand200_RFMRI.dtseries.nii'

DEFAULT_FEATURES_FILE_PATH_TEMPLATE = r'..\test_resources\%s_RFMRI_nosmoothing.dtseries.nii'
