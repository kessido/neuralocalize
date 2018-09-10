import scipy.sparse.linalg
import os

import numpy as np

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

    def __init__(self, path_to_nii_file):
        self._path = path_to_nii_file
        self.cifti, self.brain_maps = utils.cifti_utils.load_cifti_brain_data_from_file(self._path)


class Subject(object):
    """A class containing a subject and everything related to it"""

    def __init__(self, name, left_right_hemisphere_data_path='', sessions_nii_paths=[]):
        self.name = name
        self.correlation_coefficient = None
        self.features_extractor_uuid = None
        self.features = None
        self.sessions = [Session(path) for path in sessions_nii_paths]
        if left_right_hemisphere_data_path:
            self.left_right_hemisphere_data, _ = utils.cifti_utils.load_cifti_brain_data_from_file(
                left_right_hemisphere_data_path)
            self.left_right_hemisphere_data = self.left_right_hemisphere_data.transpose()
        else:
            self.left_right_hemisphere_data = None

    def load_from_directory(self, base_session_dir):
        """Loads all subject and sessions files from directory.

        Assuming sessions are ordered by filename, and the LRH data has a specific subject.

        :param path:
        :param left_right_hemisphere_data_suffix:
        :return:The subject
        """
        for session_dir in constants.SESSION_DIRS:
            path_to_session = os.path.join(base_session_dir, session_dir)
            self.sessions.append(Session(path_to_session))
        return self


def flatten_features_for_scale(x):
    return x.reshape((x.shape[0], x.shape[1] * x.shape[2]))

class Normalizer(object):
    def __init__(self):
        self.is_fit = False
        self.mean = None
        self.std = None

    def fit(self, x, dim=None):
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

        # print(np.tile(np.mean(x, dim), dim_rep.astype(dtype=int)))
        self.mean = np.mean(x, axis=dim, keepdims=True)
        self.std = np.std(x, axis=dim, ddof=1, keepdims=True)
        self.is_fit = True
        x = x - np.tile(self.mean, dim_rep.astype(dtype=int))
        x = x / np.tile(self.std, dim_rep.astype(dtype=int))

        x[np.isnan(x)] = 0
        x[np.isinf(x)] = 0
        
        x = x / np.sqrt(dim_size - 1)
        return x

    def normalize(self, x, dim=None):
        if not self.is_fit:
            raise ValueError("The Normalizer must be fitted before calling normalize!")
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

        x = x - np.tile(self.mean, dim_rep.astype(dtype=int))
        x = x / np.tile(self.std, dim_rep.astype(dtype=int))

        x = x / np.sqrt(dim_size - 1)
        # TODO(loya) add the isnan.
        return x

    @staticmethod
    def fsl_variance_normalize(y, n=30, threshold_a=2.3, threshold_b=0.001):
        k = min(n, min(y.shape) - 1)
        u, s, v = scipy.sparse.linalg.svds(y, k)
        # u = np.array([[0, 1], [1, 0]])
        # s = np.array([[1, 0], [0, 1]])
        # v = np.array([[0,1],[1,0]])

        print("u:", u)
        print("S:", s)
        print("v:", v)
        threshold = threshold_a * np.std(v, ddof=1)
        v[np.abs(v) < threshold] = 0

        stds = np.maximum(np.std(y - u @ s @ np.transpose(v), ddof=1, axis=0), threshold_b)
        return y / np.tile(stds, (y.shape[0], 1)) # this might need to be zero and not 1

    @staticmethod
    def fsl_svd(x, n):
        if x.shape[0] < x.shape[1]:
            if n < x.shape[0]:
                eigs, d = scipy.sparse.linalg.eigs(x @ x.transpose(), n)
                u = np.zeros([len(eigs), len(eigs)])
                u[np.diag_indices_from(u)] = eigs
            else:
                eigs, d = np.linalg.eig(x @ x.transpose())
                u = np.zeros([len(eigs), len(eigs)])
                u[np.diag_indices_from(u)] = eigs
                u = np.fliplr(u)
                d = np.flipud(np.fliplr(d))
            s = np.sqrt(np.abs(d))
            v = x.tranpose() @ (u @ np.diag((1 / np.diag(s))))
        else:
            if n < x.shape[1]:
                print("XTX:", np.ndim(x.transpose() @ x))
                eigs, v = scipy.sparse.linalg.eigs(x.transpose() @ x, k=n)
                d = np.zeros([len(eigs), len(eigs)])

                d[np.diag_indices_from(d)] = eigs
                print("V[:,0]:", v[:, 0])
                print("d:", d)
            else:
                print("4")
                eigs, v = np.linalg.eig(x.transpose() @ x)
                d = np.zeros([len(eigs), len(eigs)])
                d[np.diag_indices_from(d)] = eigs
                v = np.fliplr(v)
                d = np.flipud(np.fliplr(d))
            s = np.sqrt(np.abs(d))
            u = x @ (v @ np.diag((1 / np.diag(s))))
        print("u:", u)
        print("v:", v)
        print("s:", s)
        return u, s, v