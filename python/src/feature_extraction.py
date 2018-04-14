"""This code simulates the feature extraction part of the connectivity model.
"""

import numpy as np
import numpy.matlib as matlib
import sklearn.decomposition


def run_group_ica_separately(left_hemisphere_data, right_hemisphere_data, num_ic=40, N=91282):
    # TODO num_ic, N, consts: figure out and rename.
    """Runs a group ICA for each hemisphere separately.

    :param left_hemisphere_data:
    :param right_hemisphere_data:
    :param num_ic:
    :return:
    """
    # TODO(itay)
    pass


def run_group_ica_together(left_hemisphere_data, right_hemisphere_data, num_ic=50):
    # TODO num_ic, N, consts: figure out and rename.
    """Runs a group ICA for both hemispheres, to use as spatial filters.

    :param left_hemisphere_data:
    :param right_hemisphere_data:
    :param num_ic:
    :return:
    """
    # TODO(itay)
    pass


def run_dual_regression(left_right_hemisphere_data, size_of_g=91282):
    """Runs dual regression TODO(however) expand and elaborate.

    :param left_right_hemisphere_data:
    :param size_of_g:
    :return:
    """
    # TODO(loya)
    pass


def get_subcortical_parcellation(cifti_image, brain_maps):
    """Get sub-cortical parcellation using atlas definitions and current data.
    :return: (no. voxel, cortical parcellation parts)
    """

    def do_nothing_brain_map_handler(*args):
        pass

    def use_as_is_brain_map_handler(cifti_image, current_map):
        """Uses the brain map with no prepossessing
        :param cifti_image:
        :param current_map:
        :return: numpy array (no. voxels, 1), 1 if index is in part of the current part, 0 otherwise
        """
        ret = np.zeros([cifti_image.shape[1], 1])
        start_index = current_map.index_offset
        end_index = current_map.index_offset + current_map.index_count
        ret[start_index:end_index] = 1
        return ret

    def corrcoef_and_spectral_ordering(mat):
        """Implementation of reord2 + corrcoef function that was used in the matlab version
        :param mat: The data matrix
        :return: spectral ordering of the corrcoef matrix of A
        """
        mat = np.corrcoef(mat.transpose()) + 1
        ti = np.diag(np.sqrt(1. / np.sum(mat, 0)))
        W = np.matmul(np.matmul(ti, mat), ti)
        U, S, V = np.linalg.svd(W)
        S = np.diag(S)
        P = np.multiply(np.matmul(ti, np.reshape(U[:, 1], [U.shape[0], 1])), np.tile(S[1, 1], (U.shape[0], 1)))
        return P

    def half_split_using_corrcoef_and_spectral_ordering_brain_map_handler(cifti_image, current_map):
        """This split the data into 2 different clusters using corrcoef,
        spatial ordering, and positive\negative split
        :param cifti_image:
        :param current_map:
        :return: numpy array (no. voxels , 2), each vector is a 0\1 vector representing the 2 clusters
        """
        res = np.zeros([cifti_image.shape[1], 2])
        start_index = current_map.index_offset
        end_index = current_map.index_offset + current_map.index_count
        cifti_current_map_data = cifti_image[:, start_index:end_index]
        spatial_ordering = corrcoef_and_spectral_ordering(cifti_current_map_data)
        res[start_index:end_index, :] = np.hstack((spatial_ordering > 0, spatial_ordering < 0)).astype(float)
        return res

    def label_to_function(label):
        label = label.rsplit('_', 1)[0]
        labels_to_function = {
            'CIFTI_STRUCTURE_CORTEX': do_nothing_brain_map_handler,
            'CIFTI_STRUCTURE_ACCUMBENS': use_as_is_brain_map_handler,
            'CIFTI_STRUCTURE_AMYGDALA': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
            'CIFTI_STRUCTURE_BRAIN': do_nothing_brain_map_handler,
            'CIFTI_STRUCTURE_CAUDATE': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
            'CIFTI_STRUCTURE_CEREBELLUM': ica_clustering_brain_map_handler,
            'CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL': do_nothing_brain_map_handler,
            'CIFTI_STRUCTURE_HIPPOCAMPUS': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
            'CIFTI_STRUCTURE_PALLIDUM': use_as_is_brain_map_handler,
            'CIFTI_STRUCTURE_PUTAMEN': half_split_using_corrcoef_and_spectral_ordering_brain_map_handler,
            'CIFTI_STRUCTURE_THALAMUS': ica_clustering_brain_map_handler,
        }
        return labels_to_function[label]

    def ica_clustering_brain_map_handler(cifti_image, current_map):
        """This split the data into 3 parts by counting the eddect of each of the 3 first components
        in the ICA analysis on all the voxels, and determines the cluster by the one with the maximum
        connection to the voxel.
        :param cifti_image:
        :param current_map:
        :return: numpy array (no. voxels , 3), each vector is a 0\1 vector representing the 3 clusters
        """
        start_index = current_map.index_offset
        end_index = current_map.index_offset + current_map.index_count
        cifti_current_map_data = cifti_image[:, start_index:end_index]
        # todo(kess) this FastICA does not yield the same result as
        ica_Y, _, _ = sklearn.decomposition.fastica(cifti_current_map_data, 3)
        ica_Y = np.multiply(ica_Y,
                            np.tile(np.reshape(
                                np.sign(np.sum(np.sign(np.multiply(ica_Y, (np.abs(ica_Y) > 2).astype(float))), 1)),
                                (3, 1)), (1, ica_Y.shape[1])))
        res = np.zeros([cifti_image.shape[1], ica_Y.shape[0]])
        res[start_index:end_index, :] = ica_Y.transpose()
        return res

    sub_cortex_clusters = []
    for current_map in brain_maps:
        x = label_to_function(current_map.brain_structure)(cifti_image, current_map)
        if x is not None:
            print(x.shape)
            sub_cortex_clusters.append(x)
    return np.hstack(sub_cortex_clusters).transpose()


def get_semi_dense_connectome():
    """Final feature extraction (forming semi-dense connectome)
    For each subject, load RFMRI data, then load ROIs from above to calculate semi-dense connectome.

    :return:
    """
    # TODO(loya)
    pass
