import numpy as np
from utils.utils import Normalizer, fsl_glm

def test_fsl_variance_normalize():
    id = np.array([[1,0, 0],[0, 1, 0], [0, 0, 1]], dtype=float)
    res = Normalizer.fsl_variance_normalize(id, 30)
    print (res)

def test_svd():
    id = np.array([[1,0, 0],[0,1, 0], [0, 0, 1]], dtype=float)
    print ("id:", id)
    res = Normalizer.fsl_svd(id, 2)
    print (res)

def test_fsl_glm():
    x = np.array([[1,0, 0],[0,1, 0], [0, 0, 1]], dtype=float)
    y = np.array([[1,0, 0],[0,1, 0], [0, 0, 1]], dtype=float)
    ret = fsl_glm(x,y)
    print (ret)
# test_fsl_variance_normalize()
# test_svd()
test_fsl_glm()