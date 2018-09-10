import numpy as np

RFMRI_nosmoothing_filename = '_RFMRI_nosmoothing.dtseries.nii'
struct_filename = '_Struct.dtseries.nii'
filter_filename = 'ica_both_lowdim.dtseries.nii'
model_filename = 'model.pcl.gzip'
dtype = np.float32
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