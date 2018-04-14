"""This code simulates the feature extraction part of the connectivity model.
"""

import numpy as np

def run_group_ica_seperatly(left_hemisphere_data, right_hemisphere_data, num_ic=40, N=91282):
    # TODO num_ic, N, consts: figure out and rename.
    """Runs a group ICA for each henmisphere seperatly.

    :param left_hemisphere_data:
    :param right_hemisphere_data:
    :param num_ic:
    :return:
    """
    pass


def run_group_ica_together(left_hemisphere_data, right_hemisphere_data, num_ic=50):
    # TODO num_ic, N, consts: figure out and rename.
    """Runs a group ICA for both henmispheres, to use as spatial filters.

    :param left_hemisphere_data:
    :param right_hemisphere_data:
    :param num_ic:
    :return:
    """
    pass

def run_dual_regression(left_right_hemisphere_data, size_of_g=91282):
    """Runs dual regression TODO(however) expand and elaborate.

    :param left_right_hemisphere_data:
    :param size_of_g:
    :return:
    """
    pass

def get_subcortical_parcellation():
    """Get sub-cortical parcellation using atlas definitions and current data.

    :return:
    """
    pass

def get_semi_dense_connectome():
    """Final feature extraction (forming semi-dense connectome)
    For each subject, load RFMRI data, then load ROIs from above to calculate semi-dense connectome.

    :return:
    """
    pass