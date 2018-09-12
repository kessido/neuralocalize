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

    coef_mat = np.corrcoef(result, org)
    a = []
    b = []
    for i in range(len(org)):
        a.append(coef_mat[i][len(result)+i])
        b.append(pearsonr(result[i], org[i]))
    the_real_coef = coef_mat[len(result):,:len(result)]
    del result
    del org
    # print("the coeffs:", np.diagonal(the_real_coef))
    # print("with p values:", b)
    print("mean:", np.mean(the_real_coef))
    return np.mean(the_real_coef)

r_s = []
task_ids = range(1, 87)
for task in task_ids:
    print("Task:", task)
    result_path = '..\\test_resources\\result_of_ten\\AllSubjects_%s_results.dtseries.nii' % str(task).zfill(3)
    org_path = '..\\test_resources\\Tasks\\AllSubjects_%s.dtseries.nii' % str(task).zfill(3)
    r_s.append(get_pearson_corr(result_path, org_path))

r_s = np.array(r_s)
print("Mean:", np.mean(r_s))
print("STD:", np.std(r_s))
print("Min:", np.min(r_s))
print("Max:", np.max(r_s))