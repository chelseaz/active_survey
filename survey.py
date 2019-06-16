import multiprocessing as mp
import numpy as np
import time

from collections import namedtuple

from common import timing
from selector import initialize_globals

class SimulatedSurvey(object):
    def __init__(self, R, V, validation_questions,
        subgroup_list, subgroup_to_mean, subgroup_to_prec, 
        next_question_fn, cmp_criteria,
        cache_key_fn=None, free_question_mask=None,
        max_n_cutpoints=None):

        self.R = R
        self.V = V
        self.U_last = None
        self.subgroup_list = subgroup_list
        self.subgroup_to_mean = subgroup_to_mean
        self.subgroup_to_prec = subgroup_to_prec
        self.next_question_fn = next_question_fn
        self.cmp_criteria = cmp_criteria
        self.max_n_cutpoints = max_n_cutpoints

        # Questions that have been asked by the simulated survey;
        # also includes questions that shouldn't be asked (e.g. validation questions)
        # We assume self.R contains only NAs in the validation columns
        self.asked_mask = np.zeros(R.shape, dtype=bool)
        self.asked_mask[:, validation_questions] = True
        assert np.all(np.isnan(self.R[self.asked_mask]))

        # Responses collected so far by the simulated survey
        self.responses = np.empty(self.R.shape)
        self.responses[:] = np.nan

        if free_question_mask is not None:
            # reveal questions provided for free
            self.responses[:, free_question_mask] = self.R[:, free_question_mask]
            self.asked_mask[:, free_question_mask] = True

        # Save question order in a cache, if cache key function exists
        # See OfflineSelector for structure of question order 
        self.cache_key_fn = cache_key_fn
        self.cache = None
        if self.cache_key_fn is not None:
            self.cache = dict()

    # whether all the questions have been asked
    def is_complete(self):
        return (~self.asked_mask).sum() == 0

    # predictions is a filled matrix in the shape of R
    # for revealed responses, replace predictions with response values
    def override_with_revealed(self, predictions):
        predictions_with_reality = predictions.copy()
        notna_mask = ~np.isnan(self.R)
        override_mask = np.logical_and(self.asked_mask, notna_mask)
        predictions_with_reality[override_mask] = self.R[override_mask]
        return predictions_with_reality



TaskInput = namedtuple('TaskInput', [
    'user_idx', 'V', 'prior_mean', 'prior_prec', 
    'questions_asked', 'revealed_responses', 
    'next_question_fn', 'cmp_criteria'])


def ask_one_of_user(task):
    start_time = time.time()

    next_question_result = task.next_question_fn(
        task.V,
        task.prior_mean,
        task.prior_prec,
        task.questions_asked,
        task.revealed_responses,
        task.cmp_criteria
    )

    elapsed_time = time.time() - start_time
    print("User %d took %.2f seconds" % (task.user_idx, elapsed_time))

    return (task.user_idx, next_question_result)


# Question order depends on responses to previous questions,
# so it must be computed adaptively.
class OnlineSurvey(SimulatedSurvey):
    # ask one more question of all users
    @timing
    def ask_one(self, qnum):
        n, k = self.R.shape

        n_cores = mp.cpu_count()
        print("Parallelizing per-user tasks across %d cores" % n_cores)
        with mp.Pool(processes=n_cores, initializer=initialize_globals,
            initargs=(self.V, self.max_n_cutpoints)) as pool:
            # form task definitions
            # using generator instead of list comprehension
            tasks = (TaskInput(
                        user_idx=user_idx,
                        V=self.V,
                        prior_mean=self.subgroup_to_mean[self.subgroup_list[user_idx]] 
                            if self.U_last is None else self.U_last[user_idx],
                        prior_prec=self.subgroup_to_prec[self.subgroup_list[user_idx]],
                        questions_asked=self.asked_mask[user_idx],
                        revealed_responses=self.responses[user_idx],
                        next_question_fn=self.next_question_fn,
                        cmp_criteria=self.cmp_criteria
                    ) for user_idx in range(n))

            # execute tasks in parallel
            all_results = pool.imap_unordered(ask_one_of_user, tasks)

            # process results, update asked-question indicators
            for user_idx, next_question_result in all_results:
                next_question = next_question_result['question']
                self.asked_mask[user_idx, next_question] = True
                # update write-only cache if applicable
                if self.cache_key_fn is not None:
                    key = self.cache_key_fn(user_idx)
                    if key is not None:
                        question_order = self.cache.get(key)
                        if question_order is None:
                            question_order = []
                            self.cache[key] = question_order
                        question_order.append(next_question_result)

        self.responses[self.asked_mask] = self.R[self.asked_mask]


# Question order doesn't depend on responses to previous questions.
# Hence question order can be computed offline.
class OfflineSurvey(SimulatedSurvey):

    # Call this after initialization, before calling ask_one
    # epsilon is a proportion if strategy is epsilon-greedy, otherwise None
    @timing
    def precompute_order(self, question_order_fn, epsilon):
        n, k = self.R.shape
        # assume all users have been asked same number of questions already
        n_q_remaining = (~self.asked_mask[0]).sum()
        self.all_question_order = np.zeros((n, n_q_remaining), dtype=int)

        for user_idx in range(n):
            question_order = None
            if self.cache_key_fn is not None:
                # look up user in cache
                key = self.cache_key_fn(user_idx)
                if key in self.cache:
                    question_order = self.cache[key]

            if question_order is None:
                questions_asked = self.asked_mask[user_idx]
                user_prec = self.subgroup_to_prec[self.subgroup_list[user_idx]]
                question_order = question_order_fn(
                    self.V, user_prec, questions_asked, self.cmp_criteria)
                if self.cache_key_fn is not None:
                    # cache question order for user
                    self.cache[key] = question_order

            # extract and save question order
            questions_in_order = [result['question'] for result in question_order]
            if epsilon is not None:
                questions_in_order = self._epsilon_greedy_order(questions_in_order, epsilon)
            self.all_question_order[user_idx] = questions_in_order

    # scramble question order in epsilon-greedy way
    # preserve active order when not exploring
    def _epsilon_greedy_order(self, question_order, epsilon):
        remaining_question_order = question_order
        new_question_order = []
        for i in range(len(question_order)):
            p = np.random.uniform(0,1)
            if p < epsilon:
                next_q = np.random.choice(remaining_question_order)
            else:
                next_q = remaining_question_order[0]
            new_question_order.append(next_q)
            remaining_question_order.remove(next_q)

        assert len(remaining_question_order) == 0
        return new_question_order

    # ask one more question of all users
    # note qnum starts at 1
    def ask_one(self, qnum):
        n, k = self.R.shape
        all_user_idx = range(n)
        all_next_question = self.all_question_order[:, qnum-1]
        self.asked_mask[all_user_idx, all_next_question] = True
        self.responses[self.asked_mask] = self.R[self.asked_mask]

        # print()
        # print(self.asked_mask[:5])
        # print(self.responses[:5])
