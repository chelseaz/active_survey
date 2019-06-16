import numpy as np
import pandas as pd

from collections import namedtuple
from sklearn.model_selection import train_test_split

from fancyimpute import SoftImpute, MatrixFactorization, SimpleFill
from common import cmp_criteria_for, get_completer
from bpmf import BPMF
from evaluate import evaluation_loop, CVFold
from nonconjugate import OrdinalLogitPMF
from selector import OrdinalLogitActiveSelector, OfflineSelector
from survey import OnlineSurvey, OfflineSurvey

SurveyIteration = namedtuple('SurveyIteration', ['qnum', 'complete_responses', 'cache'])
SELECTION_METHODS = ['active', 'random', 'sequential', 'epsilon_greedy']
ACTIVE_SELECTION_METHODS = ['active', 'epsilon_greedy']

# Randomly split responses into training set and simulation set, respecting size constraints.
# For a given value of args.random_seed, this split is deterministic.
def make_datasets(responses, args):
    rng = np.random.RandomState(args.random_seed)
    responses_est, responses_sim = train_test_split(
        responses,
        test_size=0.5,
        shuffle=True,
        random_state=rng)

    if args.max_n_train is not None:
        responses_est = responses_est[:args.max_n_train]
        print("responses_est.shape:", responses_est.shape)
    if args.max_n_simul is not None:
        responses_sim = responses_sim[:args.max_n_simul]
        print("responses_sim.shape:", responses_sim.shape)

    # print(responses_est.index[:5])
    # print(responses_sim.index[:5])

    return (responses_est, responses_sim)


def process_responses(responses, subgroups):
    n = responses.shape[0]
    subgroup_column = subgroups.columns[0]
    combined = responses.join(subgroups, how='inner')
    R = combined.drop(subgroup_column, axis=1).as_matrix()
    subgroup_list = combined[subgroup_column].tolist()
    return n, R, subgroup_list, combined.index


# return a k x (max # values) array of response frequency counts
def compute_response_freq(responses):
    d, k = responses.shape
    freq_counts_series = [responses[col].value_counts().sort_index() for col in responses]
    max_value = max([series.index.max() for series in freq_counts_series])
    max_n_values = int(max_value)+1  # assumes min allowable value is 0
    freq_counts = np.zeros(shape=(k, max_n_values)).astype(int)
    for i in range(k):
        for value, count in freq_counts_series[i].items():
            freq_counts[i,int(value)] = count

    return freq_counts


def run(responses, subgroups, results_writer, objects_writer, args):
    questions = responses.columns.tolist()

    # compute V based on first half of users in response matrix
    # simulate survey on second half
    responses_est, responses_sim = make_datasets(responses, args)

    n_est, R_est, subgroup_est, idx_est = process_responses(responses_est, subgroups)
    n_sim, R_sim, subgroup_sim, idx_sim = process_responses(responses_sim, subgroups)

    # initialize matrix completion classes
    soft_impute = SoftImpute(max_rank=args.rank, init_fill_method="mean", verbose=False)
    mf = MatrixFactorization(rank=args.rank)
    ordinal_logit = OrdinalLogitPMF(rank=args.rank)

    response_freq = compute_response_freq(responses_est)
    ordinal_logit.set_response_freq(response_freq)

    print("Computing V using %s on %d rows" % (args.v_method, n_est))

    if args.completion_method == 'ordlogit':
        completion_result = ordinal_logit.fit_transform(R_est)
        V_est = completion_result.V
        U_est = completion_result.U

    elif args.v_method == 'row-norm':
        # ensure rows of V have unit norm
        # this is already guaranteed by SVD
        completion_result = soft_impute.fit_transform(R_est)
        V_est = completion_result.V
        U_est = np.dot(completion_result.U, completion_result.S)

        s_thresh = np.diag(completion_result.S)
        print("Thresholded singular values", s_thresh)
        print("Implied rank", (s_thresh > 0).sum())
    elif args.v_method == 'col-norm':
        # ensure columns of V have unit norm
        # But now U_est, V_est are no longer compatible
        completion_result = soft_impute.fit_transform(R_est)
        V_est = completion_result.V
        V_col_norm = np.sqrt(np.power(V_est, 2).sum(axis=0))
        V_est = V_est / V_col_norm[np.newaxis,:]
        U_est = completion_result.U
    elif args.v_method == 'mf':
        # use matrix factorization (Frobenius norm penalization) to compute U, V
        # since this corresponds to the MAP estimate in the BPMF model
        completion_result = mf.fit_transform(R_est)
        V_est = completion_result.V
        U_est = completion_result.U

    # compute empirical covariance of u_i conditional on subgroup
    subgroup_to_cov = dict()
    subgroup_to_mean = dict()
    subgroup_series = pd.Series(subgroup_est, name='subgroup')
    for subgroup, indices in subgroup_series.groupby(subgroup_series).groups.items():
        U_est_subgroup = U_est[indices,:]
        mean_subgroup = np.mean(U_est_subgroup, axis=0)
        subgroup_to_mean[subgroup] = mean_subgroup
        cov_subgroup = np.cov(U_est_subgroup, rowvar=False)
        subgroup_to_cov[subgroup] = cov_subgroup

    print("Estimated V\n", V_est.T)
    for subgroup, cov in subgroup_to_cov.items():
        print("Estimated covariance for %s\n" % subgroup, cov)

    subgroup_to_prec = {sg: np.linalg.inv(cov) for sg, cov in subgroup_to_cov.items()}

    if args.free_covariates:
        free_covariates = args.free_covariates.split(',')
        print("Free covariates:", free_covariates)
        free_covariate_mask = [col in free_covariates for col in responses]
    else:
        print("No free covariates")
        free_covariate_mask = np.repeat(False, responses.shape[1])

    print("Simulating survey using %d rows" % n_sim)

    for selection_method in args.selection_methods.split(','):
        assert selection_method in SELECTION_METHODS
        print("\nUsing %s question selection" % selection_method)
        args.selection_method = selection_method
        simulate(R_est, R_sim, V_est, subgroup_sim, idx_sim,
            subgroup_to_prec, subgroup_to_mean,
            questions, free_covariate_mask, 
            response_freq, ordinal_logit,
            results_writer, objects_writer, args)


def simulate(R_past, R, V, subgroup_list, idx,
        subgroup_to_prec, subgroup_to_mean,
        questions, free_covariate_mask, 
        response_freq, ordinal_logit,
        results_writer, objects_writer, args):

    cmp_criteria = cmp_criteria_for(args.optimality)
    epsilon = 0.05 if args.selection_method == 'epsilon_greedy' else None
    R_start_index = R_past.shape[0]
    
    if args.completion_method == 'ordlogit' and args.selection_method in ACTIVE_SELECTION_METHODS:
        ordinal_logit_active_selector = OrdinalLogitActiveSelector(V, response_freq, epsilon)
        next_question_fn = ordinal_logit_active_selector.next_question
        print("Using %s to select next question" % next_question_fn.__name__)
        
        # Cache results for a random subset of users
        n_users_cached = 100
        cached_user_idx = set(np.random.choice(R.shape[0], n_users_cached, replace=False))
        cache_key_fn = lambda user_idx: user_idx if user_idx in cached_user_idx else None
    else:
        offline_selector = OfflineSelector(args.alpha)
        next_question_fn = None  # deprecated for offline selector
        question_order_fn = getattr(offline_selector, args.selection_method + '_question_order')
        print("Using %s to compute question order" % question_order_fn.__name__)

        if question_order_fn == offline_selector.random_question_order:
            cache_key_fn = None
        else:
            # Cache results based on subgroup
            cache_key_fn = lambda user_idx: subgroup_list[user_idx]

    if args.completion_method == 'bpmf':
        completer = BPMF(V, subgroup_list, subgroup_to_mean, subgroup_to_prec, args.alpha)
    elif args.completion_method == 'ordlogit':
        completer = ordinal_logit
    else:
        completer = get_completer(args.completion_method, args.rank)


    # generator that instantiates a survey and yields predictions after each new question
    def survey_iterator(fold):
        if args.completion_method == 'ordlogit' and args.selection_method in ACTIVE_SELECTION_METHODS:
            survey = OnlineSurvey(fold.R_train, V, fold.validation_cols,
                subgroup_list, subgroup_to_mean, subgroup_to_prec, 
                next_question_fn, cmp_criteria,
                cache_key_fn=cache_key_fn,
                max_n_cutpoints=ordinal_logit_active_selector.cutpoints.shape[1])
        else:
            survey = OfflineSurvey(fold.R_train, V, fold.validation_cols,
                subgroup_list, subgroup_to_mean, subgroup_to_prec, 
                next_question_fn, cmp_criteria,
                cache_key_fn=cache_key_fn, free_question_mask=free_covariate_mask)
            survey.precompute_order(question_order_fn, epsilon)

        qnum = 0
        # include the iteration before any questions
        if args.completion_method == 'bpmf':
            complete_responses = completer.complete(survey.responses).X_filled
        else:
            # perform mean imputation
            # other completion methods have undefined behavior
            mean_imputer = SimpleFill()
            # combine R_past and simulated responses from R for matrix completion
            R_combined = np.concatenate((R_past, survey.responses), axis=0)
            completion_result = mean_imputer.fit_transform(R_combined)
            complete_responses = completion_result.X_filled[R_start_index:]
        
        yield SurveyIteration(qnum, complete_responses, survey.cache)

        # Survey question loop
        while not survey.is_complete():
            qnum += 1
            print("Expanding question bank to length", qnum)
            # ask one more question from all users
            survey.ask_one(qnum)

            if args.completion_method == 'bpmf':
                complete_responses = completer.complete(survey.responses).X_filled
            else:
                # combine R_past and simulated responses from R for matrix completion
                R_combined = np.concatenate((R_past, survey.responses), axis=0)
                completion_result = completer.fit_transform(R_combined)
                complete_responses = completion_result.X_filled[R_start_index:]
                survey.U_last = completion_result.U

            yield SurveyIteration(qnum, complete_responses, survey.cache)


    # Compute active ordering on full question set, save results
    if not args.skip_cache:
        cache = simulation_loop(R, survey_iterator)
        sim_key = "%s_%s" % (args.sim_uuid, args.selection_method)
        objects_writer.save_objects(sim_key, dict(
            args=args,
            cache=cache,
            V=V,
            subgroup_to_mean=subgroup_to_mean,
            subgroup_to_prec=subgroup_to_prec
        ))

    # Run survey on different validation sets
    # Separate from simulation_loop since not all questions may appear
    if not args.skip_eval:
        evaluation_loop(R, survey_iterator, results_writer, args, questions,
            eval_blacklist=free_covariate_mask)


def simulation_loop(R, survey_iterator):
    no_validation_fold = CVFold(
        name="full_data", R=R,
        validation_mask=np.zeros(R.shape).astype(bool)
    )
    print("Simulating survey on full data")
    cache = None
    for iteration in survey_iterator(no_validation_fold):
        cache = iteration.cache
    return cache
