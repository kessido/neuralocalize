import os

import numpy as np

import utils.cifti_utils


def create_dir(path):
    """If the dir in path does not exist, create it.

    :param path: The dir path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def remove_elements_from_list(input_list, elements):
    """ Remove all elements from list that are also on another list.

    :param list: The list to remove from.
    :param elements: The elements to remove.
    :return: The list without elements.
    """
    return list(set(input_list) - set(elements))


def add_ones_column_to_matrix(mat):
    """ Add a column of 1's
    Usually needed for linear algebra.

    :param mat: The original matrix
    :return: The matrix with another 1's column, as it's first column.
    """
    shape = list(mat.shape)
    shape[1] += 1
    res = np.ones(shape)
    res[1:] = mat
    return res


def fsl_glm(x, y):
    """Translation of the MATLAB fsl_glm method into python.

    Args:
        x: data
        y: labels

    Returns:
        the t-coefficient of the data.
    """
    beta = np.dot(np.linalg.pinv(x), y)
    r = y - np.dot(x, beta)
    dof = np.size(y, 0) - np.linalg.matrix_rank(x)

    sigma_sq = np.sum(r ** 2, axis=0) / dof

    grot = np.diag(np.linalg.inv((x.transpose().dot(x))))
    varcope = grot.reshape([len(grot), 1]).dot(sigma_sq.reshape([1, len(sigma_sq)]))
    t = beta / np.sqrt(varcope)
    return t


class Session(object):
    """A class representing a session"""

    def __init__(self, path_to_nii_file):
        self._path = path_to_nii_file

    @property
    def cifti(self):
        cifti, _ = self.cifti_brain_maps
        return cifti

    @property
    def brain_maps(self):
        _, brain_maps = self.cifti_brain_maps
        return brain_maps

    @property
    def cifti_brain_maps(self):
        return utils.cifti_utils.load_nii_brain_data_from_file(self._path)


class Subject(object):
    """A class containing a subject and everything related to it"""

    def __init__(self, name, left_right_hemisphere_data_path='', sessions_nii_paths=[]):
        self.name = name
        self.sessions = [Session(path) for path in sessions_nii_paths]
        if left_right_hemisphere_data_path:
            self.left_right_hemisphere_data, _ = utils.cifti_utils.load_nii_brain_data_from_file(
                left_right_hemisphere_data_path)
            self.left_right_hemisphere_data = self.left_right_hemisphere_data.transpose()
        else:
            self.left_right_hemisphere_data = None

    def load_from_directory(self, path, left_right_hemisphere_data_suffix):
        """Loads all subject and sessions files from directory.

        Assuming sessions are ordered by filename, and the LRH data has a specific subject.

        :param path:
        :param left_right_hemisphere_data_suffix:
        :return:
        """
        ordered_files = os.listdir(path).sort()
        for file in ordered_files:
            full_path = os.path.join(path, file)
            if file.endswith(left_right_hemisphere_data_suffix):
                self.left_right_hemisphere_data, _ = utils.cifti_utils.load_nii_brain_data_from_file(full_path)
                self.left_right_hemisphere_data = self.left_right_hemisphere_data.transpose()
            else:
                self.sessions.append(Session(full_path))
