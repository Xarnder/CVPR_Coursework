import numpy as np

def cvpr_compare(F1, F2):
    f1 = np.asarray(F1).ravel()
    f2 = np.asarray(F2).ravel()
    return np.linalg.norm(f1 - f2, ord=2)

