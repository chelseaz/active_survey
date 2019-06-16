import numpy as np

def clobber(matrix, clobber_mask):
    clobbered = matrix.copy()
    clobbered[clobber_mask] = np.nan
    return clobbered


class MatrixSparsity(object):
    """Uniformly punch holes in a matrix"""
    def __init__(self, sparsity_level):
        self.sparsity_level = sparsity_level

    def get_clobber_mask(self, matrix):
        return np.random.random(size=matrix.shape) < self.sparsity_level


class UserSparsity(object):
    """Downsample proportion of responses per user"""
    def __init__(self, sparsity_level):
        self.sparsity_level = sparsity_level

    def get_clobber_mask(self, matrix):
        clobber_mask = np.zeros(matrix.shape)
        n_questions = matrix.shape[1]
        clobber_per_user = int(np.round(n_questions * self.sparsity_level))
        for user_index in range(matrix.shape[0]):
            clobber_indices = np.random.choice(n_questions, size=clobber_per_user, replace=False)
            clobber_mask[user_index, clobber_indices] = 1
        return clobber_mask.astype(bool)


# To compare these methods:
# us = UserSparsity(0.5)
# ms = MatrixSparsity(0.5)
# us.get_clobber_mask(np.zeros((10, 6))).astype(int)
# ms.get_clobber_mask(np.zeros((10, 6))).astype(int)
# us.get_clobber_mask(np.zeros((10, 6))).sum(axis=1)