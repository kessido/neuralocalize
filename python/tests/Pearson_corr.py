import numpy as np
import utils.cifti_utils
from scipy.stats.stats import pearsonr


def get_pearson_corr(result_path, org_path):
    """Calculate
    :param result - an array that represent nii file of the python result.
    :param org - an array that represent nii file of the original task.
    :return: The correlation between -1 and 1.
    """
    result,_ = utils.cifti_utils.load_cifti_brain_data_from_file(result_path)
    org,_ = utils.cifti_utils.load_cifti_brain_data_from_file(org_path)
    org = np.resize(org, result.shape)

    result = result#.flatten()
    org = org#.flatten()

    # result = np.array([[1,1,0.9,4]])
    # org = np.array([[1, 0.9, 1,4.1]])
    # for line in range(org.shape[0]):
    #     print("line ", line)
    #     print(pearsonr(result[line], org[line]))
    coef_mat = np.corrcoef(result, org)
    a = []
    b = []
    for i in range(len(org)):
        a.append(coef_mat[i][len(result)+i])
        b.append(pearsonr(result[i], org[i]))
    the_real_coef = coef_mat[len(result):,:len(result)]
    print("a:", a)
    print("b:", b)
    print("the real coef:", np.diagonal(the_real_coef))
    return



result_path = 'AllSubjects_001_results.dtseries.nii'
org_path = 'AllSubjects_001.dtseries.nii'
print(get_pearson_corr(result_path, org_path))
