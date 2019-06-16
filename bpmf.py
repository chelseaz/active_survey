import numpy as np

from common import CompletionResult


class BPMF(object):
    """
    Implementation of Bayesian probabilistic matrix factorization
    using fixed V and empirical Bayes prior for rows of U, based on subgroup. 
    Predictions are obtained using the MAP estimate of U.
    """

    def __init__(self, 
        V,  # fixed  
        subgroup_list,  # subgroup membership of each row in response matrix
        subgroup_to_mean,  # mapping of subgroup to prior mean
        subgroup_to_prec,  # mapping of subgroup to prior precision
        alpha  # precision of response given user, item factors
    ):
        self.V = V
        self.subgroup_list = np.array(subgroup_list)
        self.subgroup_to_mean = subgroup_to_mean
        self.subgroup_to_prec = subgroup_to_prec
        self.alpha = alpha

    def complete(self, X):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """

        # X_filled takes care of the indicators by setting NA to 0
        X_filled = np.nan_to_num(X)

        # construct U_map in place

        # first term
        U_map = self.alpha * np.matmul(X_filled, self.V.T)

        # add second term by subgroup
        for subgroup in np.unique(self.subgroup_list):
            cur_subgroup_mask = (self.subgroup_list == subgroup)
            prior_mean = self.subgroup_to_mean[subgroup]
            prior_prec = self.subgroup_to_prec[subgroup]

            # add prior product to each row in the subgroup
            U_map[cur_subgroup_mask] += np.matmul(prior_mean, prior_prec)

        # multiply by posterior variance
        # make no assumptions about common question orders across users
        # in other words, assume a random strategy
        # iterating through users will incur a performance hit
        # TODO: vectorize this
        for i in range(U_map.shape[0]):
            prev_questions = ~np.isnan(X[i])
            V_prev = self.V[:, prev_questions]
            prior_prec = self.subgroup_to_prec[self.subgroup_list[i]]
            posterior_prec = prior_prec + self.alpha * np.matmul(V_prev, V_prev.T)
            posterior_var = np.linalg.inv(posterior_prec)

            U_map[i] = np.matmul(U_map[i], posterior_var)

        return CompletionResult(
            X_filled=np.matmul(U_map, self.V),
            U=U_map,
            V=self.V,
            S=None,
            rank=self.V.shape[0]
        )