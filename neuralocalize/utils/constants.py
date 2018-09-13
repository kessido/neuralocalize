import numpy as np
import pkg_resources

DTYPE = np.float64
ICA_THRESHOLD_CONST = 0.005

PATH_TO_SESSION = "MNINonLinear/Results"
SESSION_IDS = [('1', 'LR'), ('1', 'RL'), ('2', 'LR'), ('2', 'RL')]

PATH_TO_SESSIONS = "MNINonlinear/Results"
SESSION_NAME_TEMPLATE = "rfMRI_REST%s_%s/rfMRI_REST%s_%s_Atlas_MSMAll_hp2000_clean.dtseries.nii"
SESSION_DIRS = [SESSION_NAME_TEMPLATE % (num, side, num, side) for num, side in SESSION_IDS]


DEFAULT_ICA_BOTH_RESULT_PATH = pkg_resources.resource_filename(__name__, "resources/ica_both_lowdim.dtseries.nii")
DEFAULT_ICA_SEPERATED_RESULT_PATH = pkg_resources.resource_filename(__name__, "resources/ica_LR_MATCHED.dtseries.nii")
DEFAULT_STRUCTURE_ICA_RESULT_PATH = pkg_resources.resource_filename(__name__, "resources/SC_clusters.dtseries.nii")
EXAMPLE_FILE_PATH = pkg_resources.resource_filename(__name__, "resources/example.dtseries.nii")