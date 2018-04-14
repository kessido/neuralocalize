import pytest
import nibabel as nib
import numpy as np

def get_matlab_matrix_as_numpy(nii_path):
    """Converts nii object into numpy matrix.
    :param nii_path: A path to the nii file.
    :return: numpy matrix with the same data.
    """
    nib_data = nib.load(nii_path)
    return np.array(nib_data)  # TODO(loya) make sure np.array and not np.matrix, potential bug.



# TODO(loya) when we have a list of methods and files, create a decorator and run.
def abstract_test(method_to_test, nii_path):
    """Test pattern for the module: runs the method and checks that the result is the same.
    :param method_to_test: The method we wish to test
    :param nii_path: The path holding the expected matlab matrix.
    """
    actual_output = method_to_test()  # TODO(loya) handle params if needed.
    expected_output = get_matlab_matrix_as_numpy(nii_path)
    assert np.allclose(actual_output, expected_output)
