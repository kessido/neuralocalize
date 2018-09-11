import os
import warnings

import numpy as np
import scipy.sparse.linalg

import constants
import utils.cifti_utils


def create_dir(path):
    """If the dir in path does not exist, create it.

    :param path: The dir path.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def remove_elements_from_list(input_list, elements):
    """ Remove all elements from list that are also on another list.


    :param input_list: The list to remove from.
    :param elements: The elements to remove.
    :return: The list without elements.
    """
    input_set = set(input_list)
    if len(input_list) != len(input_set):
        warnings.warn("In remove_elements_from_list in utils.utils: " +
                      "the input list contains duplicates which will be removed, "
                      "if this is not the desired behavior please implement this "
                      "yourself.")

    return list(input_set - set(elements))


def add_ones_column_to_matrix(mat):
    """ Add a column of 1's
    Usually needed for linear algebra.

    :param mat: The original matrix
    :return: The matrix with another 1's column, as it's first column.
    """
    shape = list(mat.shape)
    shape[1] += 1
    res = np.ones(shape)
    res[:, 1:] = mat
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
    t[np.isnan(t)] = 0
    return t


def fsl_demean(x, dim=None):
    """Implementation of the MATLAB fsl_demean method into python

    :param x: The data to demean.
    :param dim: The dim index to demean by.
    :return: The demeaned data.
    """
    if dim is None:
        dim = 0
        if x.shape[0] > 1:
            dim = 0
        elif x.shape[1] > 1:
            dim = 1

    dims = x.shape
    dim_size = dims[dim]
    dim_rep = np.ones([len(dims)])
    dim_rep[dim] = dim_size
    mean = np.mean(x, dim, keepdims=True)
    x = x - np.tile(mean, dim_rep.astype(dtype=int))
    return x


class Session(object):
    """A class representing a session"""

    def __init__(self, cifti_path):
        """Initialize the session instance from cifti file path.

        :param cifti_path: The path to the cifti file of the session.
        """
        self.cifti, self.brain_maps = utils.cifti_utils.load_cifti_brain_data_from_file(cifti_path)


class Subject(object):
    """A class containing a subject and everything related to it"""

    def __init__(self, name, left_right_hemisphere_data_path=None, sessions_nii_paths=None):
        """Initialize the Subject instance.

        :param name: The name of the subject
            (Under the human connectome project it usually a 6 digit anonymous name).
        :param left_right_hemisphere_data_path:
        :param sessions_nii_paths:
        """
        self.name = name
        self.correlation_coefficient = None
        self.features_extractor_uuid = None
        self.features_before_scaling = None

        if sessions_nii_paths is not None:
            self.sessions = [Session(path) for path in sessions_nii_paths]
        else:
            self.sessions = []

        self.left_right_hemisphere_data = None
        if left_right_hemisphere_data_path is not None:
            self.left_right_hemisphere_data, _ = utils.cifti_utils.load_cifti_brain_data_from_file(
                left_right_hemisphere_data_path)
            self.left_right_hemisphere_data = self.left_right_hemisphere_data.transpose()

    def load_from_directory(self, base_session_dir):
        """Loads all subject and sessions files from directory.

        Assuming sessions are ordered by filename, and the LRH data has a specific subject.

        :param base_session_dir: the path to load the session from.
        """
        for session_dir in constants.SESSION_DIRS:
            path_to_session = os.path.join(base_session_dir, session_dir)
            self.sessions.append(Session(path_to_session))
        return self


class Normalizer(object):
    """A class for normalizing features"""

    def __init__(self, dim=None):
        """Initialize a new Normalize instance to save scaling.

        :param dim: The dimension to normalize by.
        """
        self.is_fit = False
        self.mean = None
        self.std = None
        self.dim = dim

    def fit(self, x, y=None):
        """Fit the normalizer to the data.

        :param x: The data.
        :param y: for compatibility.
        """
        dim = self.dim
        if dim is None:
            dim = 0
            if x.shape[0] > 1:
                dim = 0
            elif x.shape[1] > 1:
                dim = 1

        dims = x.shape
        dim_size = dims[dim]
        dim_rep = np.ones([len(dims)])
        dim_rep[dim] = dim_size

        self.dim = dim
        self.mean = np.mean(x, axis=dim, keepdims=True)
        self.std = np.std(x, axis=dim, ddof=1, keepdims=True)
        self.is_fit = True

    def fit_transform(self, x, y=None):
        """Fit the normalizer to the data, and then return the normalized data.

        :param x: The data.
        :param y: For compatibility.
        :return: The normalized data.
        """
        self.fit(x)
        return self.transform(x)

    def transform(self, x):
        if not self.is_fit:
            raise ValueError("The Normalizer must be fitted before calling normalize!")

        dim = self.dim
        dims = x.shape
        dim_size = dims[dim]
        dim_rep = np.ones([len(dims)])
        dim_rep[dim] = dim_size

        x = x - np.tile(self.mean, dim_rep.astype(dtype=int))
        x = x / np.tile(self.std, dim_rep.astype(dtype=int))

        x = x / np.sqrt(dim_size - 1)
        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        return x

    @staticmethod
    def fsl_variance_normalize(y, n=30, threshold_a=2.3, threshold_b=0.001):
        k = min(n, min(y.shape) - 1)
        u, s, v = scipy.sparse.linalg.svds(y, k)
        s = scipy.linalg.diagsvd(s, k, k)
        v = v.transpose()

        threshold = threshold_a * np.std(v, ddof=1)
        v[np.abs(v) < threshold] = 0

        stds = np.maximum(np.std(y - u @ s @ np.transpose(v), ddof=1, axis=0), threshold_b)
        return y / np.tile(stds, (y.shape[0], 1))  # this might need to be zero and not 1
