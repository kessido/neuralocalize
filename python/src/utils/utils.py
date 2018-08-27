import os
import numpy as np

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def remove_list_from_list(list1, list2):
    return list(set(list1) - set(list2))

def fsl_glm(x, y):
    """Translation of the MATLAB fsl_glm method into python.

    Args:
        x: data
        y: labels

    Returns:
        the t-coefficient of the data.
    """
    beta = np.dot(np.linalg.pinv(x) , y)
    r = y - np.dot(x, beta)
    dof = np.size(y, 0) - np.linalg.matrix_rank(x)

    sigma_sq = np.sum(r**2, axis=0) / dof

    grot = np.diag(np.linalg.inv((x.transpose().dot(x))))
    varcope = grot.reshape([len(grot), 1]).dot(sigma_sq.reshape([1, len(sigma_sq)]))
    t = beta / np.sqrt(varcope)
    return t