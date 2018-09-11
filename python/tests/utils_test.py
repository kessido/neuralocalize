import numpy as np
import scipy.signal
import scipy.io
from utils.utils import Normalizer, fsl_glm
import utils.cifti_utils

def test_fsl_variance_normalize():
    id = np.array([[1,1, 1,4],[2, 2, 2,0], [0, 0, 3,0]], dtype=float)
    res = Normalizer.fsl_variance_normalize(id, 30)
    print (res)

def test_fsl_glm():
    y = np.array([[1,1, 1],[2,2, 2], [3, 3, 3]], dtype=float)
    x = np.array([[1,0, 0],[0,1, 0], [0, 0, 1]], dtype=float)
    x = scipy.io.loadmat('..\\test_resources\T.mat')['T']
    y, _ = utils.cifti_utils.load_cifti_brain_data_from_file('..\\test_resources\data.dtseries.nii')
    y = y.transpose()
    print("x[0]", x[0])
    print("y[0]", y[0])
    ret = fsl_glm(x.transpose(),y.transpose())
    print (ret)


def test_detrend():
    x = np.array([[1, 1, 1],[2,2, 2], [3, 3, 3]], dtype=float)
    print (scipy.signal.detrend(x, type='constant', axis=0))


test_detrend()