import edward as ed
import gc
import numpy as np
import tensorflow as tf

from memory_profiler import profile

from common import compute_opt_criteria
from nonconjugate import get_cutpoints, OrdinalLogitRegression, OrdinalLogitSingleObs


# Sadly we need globals to initialize state within worker processes...
# See
# https://stackoverflow.com/questions/10117073/how-to-use-initializer-to-set-up-my-multiprocess-pool
# https://thelaziestprogrammer.com/python/multiprocessing-pool-a-global-solution
# To conserve memory, create TF ops for item selection only once
# and sequester them in a separate TF graph
selection_graph = None
ol_single_obs = None
def initialize_globals(V, max_n_cutpoints):
    d, k = V.shape
    global selection_graph
    global ol_single_obs

    selection_graph = tf.Graph()
    with selection_graph.as_default():
        with tf.Session() as sess:
            ol_single_obs = OrdinalLogitSingleObs(d, max_n_cutpoints)


class OrdinalLogitActiveSelector(object):
    def __init__(self, V, response_freq, epsilon, min_obs_for_map=1):
        self.epsilon = epsilon
        self.min_obs_for_map = min_obs_for_map

        # length-K vector representing maximum response value per question
        # assumes response values start at 0
        self.max_level_vec = [np.max(np.where(row)) for row in response_freq > 0]

        with tf.Session() as sess:
            cutpoints = get_cutpoints(response_freq).eval()
        self.cutpoints = cutpoints

    # Item factors considered fixed, ordinal logit likelihood
    # Select item that maximizes expected Fisher information of user factors
    # at the MAP estimate of user factors where available, 
    # or at the prior mean of user factors if data is insufficient.
    # @profile
    def next_question(self,
        V, prior_mean, prior_prec, questions_asked, revealed_responses, cmp_criteria):

        unobs_idx, = np.where(~questions_asked)

        if self.epsilon is not None:
            # perform epsilon-greedy selection
            p = np.random.uniform(0,1)
            if p < self.epsilon:
                return dict(question=np.random.choice(unobs_idx))

        d, k = V.shape
        responses_present = ~np.isnan(revealed_responses)  # subset of TRUE entries in questions_asked
        
        # # perform MAP estimation if enough responses present
        # if np.sum(responses_present) >= self.min_obs_for_map:
        #     # perform in default graph so operations can get reset afterward
        #     g = tf.get_default_graph()
        #     print("START: Number of ops in default graph", len(g.get_operations()))

        #     with tf.Session() as sess:
        #         ol_regression = OrdinalLogitRegression(k, d, self.cutpoints.shape[1])
        #         u_map = ol_regression.map_estimate(
        #             V_data=V.T, 
        #             R_data=revealed_responses[responses_present], 
        #             I_data=responses_present,
        #             cutpoints_data=self.cutpoints, 
        #             prior_mean_data=prior_mean, 
        #             prior_prec_data=prior_prec)
        #         del ol_regression

        #     print("Resetting default graph")
        #     tf.reset_default_graph()
        #     tf.keras.backend.clear_session()  # not clear what this does
        #     g = tf.get_default_graph()
        #     print("END: Number of ops in default graph", len(g.get_operations()))

        # else:
        u_map = prior_mean  # kind of a misnomer

        # print("prior mean is", prior_mean)
        # print("u_map is", u_map)

        # isolate operations for item selection in separate graph to manage memory
        with selection_graph.as_default():
            with tf.Session() as sess:
                # Compute observed Fisher information from existing responses
                obs_fisher = prior_prec.copy()
                obs_idx, = np.where(responses_present)
                for i in obs_idx:
                    cutpoints_i = self.cutpoints[i][np.newaxis,:]
                    obs_fisher += ol_single_obs.new_point_info(
                        V.T[[i]], u_map, revealed_responses[[i]], cutpoints_i)

                # Choose next question from unasked pool
                # TODO: natural place to introduce randomness for efficiency gains,
                # by limiting choices to a random subset of unasked questions
                Lambda_prev = obs_fisher
                best_new, best_opt_criteria = None, None
                for i in unobs_idx:
                    v_cand = V.T[[i]]
                    max_level = self.max_level_vec[i]
                    cutpoints_i = self.cutpoints[i][np.newaxis,:]
                    Lambda_cand = Lambda_prev + ol_single_obs.new_point_info_expected(
                        v_cand, u_map, max_level, cutpoints_i)
                    opt_criteria = compute_opt_criteria(Lambda_cand)
                    
                    if best_opt_criteria is None or cmp_criteria(opt_criteria, best_opt_criteria) < 0:
                        best_new = i
                        best_opt_criteria = opt_criteria

                result = dict(question=best_new)
                result.update(best_opt_criteria._asdict())

        print("Result", result)
        return result

class OfflineSelector(object):
    def __init__(self, alpha):
        self.alpha = alpha

    # Given 
    # - question factors v_1, \ldots, v_k as V (d x k matrix)
    # - prior precision for user factors
    # - already asked questions as true/false array
    # - a comparator based on some optimality criterion
    # Return a sequence of dicts containing
    # - the optimal next question for active learning
    # - optimality criteria
    def active_question_order(self, V, user_prec, prev_questions, cmp_criteria):
        d, k = V.shape
        assert prev_questions.size == k
        V_prev = V[:, prev_questions]
        Lambda_prev = user_prec.copy() / self.alpha  # prior precision
        Lambda_prev += np.dot(V_prev, V_prev.T)

        active_order = []
        remaining_questions = set(np.where(~prev_questions)[0])
        while len(remaining_questions) > 0:
            # choose next question
            best_question = None
            best_opt_criteria = None
            for i in remaining_questions:
                # TODO: more efficient way to compute eigenvalues of matrix sum?
                v_i = V[:, i]
                Lambda = Lambda_prev + np.outer(v_i, v_i)
                opt_criteria = compute_opt_criteria(Lambda)

                # minimize optimality criterion
                if best_opt_criteria is None or cmp_criteria(opt_criteria, best_opt_criteria) < 0:
                    best_question = i
                    best_opt_criteria = opt_criteria

            result = dict(question=best_question)
            result.update(best_opt_criteria._asdict())
            active_order.append(result)

            v_best = V[:, best_question]
            Lambda_prev += np.outer(v_best, v_best)
            remaining_questions.remove(best_question)

        return active_order

    def random_question_order(self, V, user_prec, prev_questions, cmp_criteria):
        remaining_questions, = np.where(~prev_questions)
        return [dict(question=q) for q in np.random.permutation(remaining_questions)]

    def sequential_question_order(self, V, user_prec, prev_questions, cmp_criteria):
        # don't assume previous questions are sequential
        remaining_questions, = np.where(~prev_questions)
        # this should be in ascending order, but sort to be safe
        return [dict(question=q) for q in np.sort(remaining_questions)]

    def epsilon_greedy_question_order(self, V, user_prec, prev_questions, cmp_criteria):
        # randomization is handled after active question order is computed
        return self.active_question_order(V, user_prec, prev_questions, cmp_criteria)
