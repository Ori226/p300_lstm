import numpy as np


def myMatrixConcat(source_mat, target_mat):
    if source_mat is None:
        return_value = np.asarray(target_mat)
    else:
        return_value = np.vstack((source_mat, np.asarray(target_mat)))

    return return_value

