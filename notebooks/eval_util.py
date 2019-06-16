import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns

import os, sys
from IPython.display import display, HTML
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import compute_opt_criteria
from evaluate import Metrics
from bpmf import BPMF


def get_active_sim_keys(sim_objects, rank, optimality_type, v_method, alpha):
    def include(args):
        return args.selection_method == 'active' and \
            args.rank == rank and \
            args.optimality == optimality_type and \
            args.v_method == v_method and \
            args.alpha == alpha and args.eval_method == 'sparsify'

    return [sim_key for sim_key, o in sim_objects.items() if include(o['args'])]

def extract_from_cache(sim_objects, sim_keys, key, subgroup):
    print("Extracting cache for {:d} simulations".format(len(sim_keys)))
    sim_key_to_values = dict()
    for sim_key in sim_keys:
        o = sim_objects[sim_key]
        cache = o['cache']
        subgroup_result = cache[subgroup]
        values = [qinfo[key] for qinfo in subgroup_result]
        sim_key_to_values[sim_key] = values
    return sim_key_to_values

# compute optimality criteria according to random strategy, averaged over n_reps
def compute_random_seq(sim_objects, sim_key, opt_measure, alpha, subgroup,
    length=None, n_reps=100):

    V = sim_objects[sim_key]['V']
    prior_prec = sim_objects[sim_key]['subgroup_to_prec'][subgroup]
    d, k = V.shape
    if length is None: 
        length = k
    all_seq = []
    
    for i in range(n_reps):
        # generate a random question order
        question_order = np.random.permutation(length)
        opt_measure_seq = []
        Lambda = prior_prec.copy() / alpha  # prior precision
        for question in question_order:
            v_i = V[:, question]
            Lambda += np.outer(v_i, v_i)
            opt_criteria = compute_opt_criteria(Lambda)
            opt_measure_seq.append(getattr(opt_criteria, opt_measure))
        all_seq.append(opt_measure_seq)
    
    mean_seq = np.vstack(all_seq).mean(axis=0)
    return mean_seq

opt_measure_for_type = dict(A='trace', D='logdet', E='max_eigval')

def plot_objective(sim_objects, rank, optimality_type, v_method, alpha, subgroup='all',
    yscale='linear'):
    
    opt_measure = opt_measure_for_type[optimality_type]
    sim_keys = get_active_sim_keys(sim_objects, rank, optimality_type, v_method, alpha)
    
    # plot optimality measure
    sim_key_to_opt_seq = extract_from_cache(sim_objects, sim_keys, opt_measure, subgroup)
    
    sns.set_style("ticks")
    plt.rcParams["figure.figsize"] = (6, 5)
    plt.figure()

    labeled = False
    for sim_key, opt_seq in sim_key_to_opt_seq.items():
        active_opt_seq = np.array(opt_seq)
        active_opt_seq[active_opt_seq > 1e6-1] = np.nan
        label = 'active' if not labeled else None
        labeled = True
        plt.plot(range(1,len(active_opt_seq)+1), active_opt_seq, marker=None, color='r', label=label)
        
    labeled = False
    for sim_key, _ in sim_key_to_opt_seq.items():
        random_opt_seq = compute_random_seq(sim_objects, sim_key, opt_measure, alpha, subgroup,
            length=len(active_opt_seq))
        random_opt_seq[random_opt_seq > 1e6-1] = np.nan
        label = 'random' if not labeled else None
        labeled = True
        plt.plot(range(1,len(random_opt_seq)+1), random_opt_seq, marker=None, color='k', label=label)
    
    plt.xlabel('Questions')
    plt.ylabel(opt_measure.capitalize())
    plt.yscale(yscale)
    subgroup_str = 'all users' if subgroup == 'all' else subgroup
    plt.title("Active learning objective, rank={:d}, alpha={:d}, {}".format(rank, alpha, subgroup_str), y=1.02)
    plt.legend()



cat_q_pattern = re.compile('^([^_]*)_(.*)$')

def label_for_question(question, question_to_label):
    label = question_to_label.get(question)
    if label is None:
        m = re.search(cat_q_pattern, question)
        question_prefix, cat_value = m.groups()
        label = "{} - {}".format(question_to_label.get(question_prefix), cat_value)
    
    return label

def display_V(sim_objects, questions, question_to_label,
              rank, optimality_type, v_method, alpha, 
              max_sims=1, filename=None, aspect=0.4):
    
    n_questions = len(questions)
    labels = [label_for_question(q, question_to_label) for q in questions]
    sim_keys = get_active_sim_keys(sim_objects, rank, optimality_type, v_method, alpha)
    for sim_key in sim_keys[:max_sims]:
        print(sim_key)
        V = sim_objects[sim_key]['V']
        questions = pd.Series([q.replace('_', '-') for q in questions], name='Question')
        pc_labels = ['PC{:d}'.format(i) for i in range(1, rank+1)]
        V_readable = pd.DataFrame(V, columns=questions, index=pc_labels).T
        V_readable['Text'] = labels
#         with pd.option_context('display.max_rows', None):
#             display(V_readable)

        if filename is not None:
            uuid_suffix = '-%s' % sim_key.split('-')[0][-4:]
            uuid_filename = filename.replace('.csv', '{}.csv'.format(uuid_suffix))
            V_readable.to_csv(uuid_filename, float_format='%.2f')

        V_df = pd.melt(V_readable.reset_index(), id_vars=['Question', 'Text'], var_name='PC')\
                 .sort_values(by=['PC', 'Text'], ascending=[True, False])
#         display(V_df.head())

        sns.set(font_scale=1.5)
        sns.set_style("ticks")
        g = sns.FacetGrid(V_df, col="PC", size=n_questions/3.5, aspect=aspect, despine=True)
        g = g.map(plt.barh, "Text", "value", color='gray')\
             .set_axis_labels("", "")


def plot_V_2d(ax, V, labels, colors, title):
    d, k = V.shape
    ax.quiver(np.zeros(k), np.zeros(k), V[0], V[1], color=colors, angles='xy', scale_units='xy', scale=1)
    for i, label in enumerate(labels):
        ax.text(V[0,i], V[1,i], label, size='large', weight='bold')
    xymax = np.max(np.abs(V))
    ax.set_xlim(-xymax, xymax)
    ax.set_ylim(-xymax, xymax)
    ax.set_title(title, y=1.05)

def compute_question_ranks(question_order, n_questions):
    question_to_rank = dict((q, o+1) for o, q in enumerate(question_order))
    rank_list = [question_to_rank.get(q) for q in range(n_questions)]
    return [np.nan if rank is None else rank for rank in rank_list]

def plot_question_order(sim_objects, questions, rank, optimality_type, v_method, alpha, subgroup='all',
    max_plots=4, max_labels=None, plot_position_only=False):

    if max_labels is None: 
        max_labels = rank+1

    sim_keys = get_active_sim_keys(sim_objects, rank, optimality_type, v_method, alpha)
    sim_keys = sim_keys[:max_plots]
    
    sns.set_style("white")
    ncols = min(2, max_plots)
    nrows = int(np.ceil(len(sim_keys) / ncols))
    plt.rcParams["figure.figsize"] = (6*ncols, 6*nrows)
    fig, axarr = plt.subplots(nrows=nrows, ncols=ncols)

    if plot_position_only:
        sim_key_to_question_order = {sim_keys[0]: []}
    else:
        sim_key_to_question_order = extract_from_cache(sim_objects, sim_keys, 'question', subgroup)
    
    for i, (sim_key, question_order) in enumerate(sim_key_to_question_order.items()):
        V = sim_objects[sim_key]['V']
        
        row = int(np.floor(i/ncols))
        col = i % ncols
        if nrows == ncols == 1:
            ax = axarr
        elif nrows == 1:
            ax = axarr[col]
        else:
            ax = axarr[row][col]

        if plot_position_only:
            plot_V_2d(ax, V, labels=[], colors='gray', title="")
        else:
            # uuid_suffix = '-%s' % sim_key.split('-')[0][-4:]
            order_list = compute_question_ranks(question_order, len(questions))
            order_labels = ["" if order is np.nan or order > max_labels else str(order) for order in order_list]
            colors = plt.cm.Blues(1 - np.array(order_list) / len(order_list))        
            plot_V_2d(ax, V, labels=order_labels, colors=colors, title="")
    
    subgroup_str = 'all users' if subgroup == 'all' else subgroup
    plt.suptitle("Questions in latent space, rank={:d}, alpha={:d}, {}".format(rank, alpha, subgroup_str), fontsize=16)

    plt.subplots_adjust(left=0, right=1, bottom=0, top=0.9, wspace=0.2, hspace=0.2)


def plot_question_ranks(sim_objects, questions, rank, optimality_type, v_method, alpha, subgroup='all', 
    show_question_labels=False, question_to_label=None, show_max_questions=None,
    max_label_length=80, sort_by_rank=True):

    n_questions = len(questions)
    sim_keys = get_active_sim_keys(sim_objects, rank, optimality_type, v_method, alpha)
    sim_key_to_question_order = extract_from_cache(sim_objects, sim_keys, 'question', subgroup)
    all_question_ranks = np.vstack([compute_question_ranks(question_order, n_questions) 
        for question_order in sim_key_to_question_order.values()])
    
    # replace NA ranks with last rank (number of questions)
    all_question_ranks[np.isnan(all_question_ranks)] = n_questions
    avg_question_ranks = np.median(all_question_ranks, axis=0)

    avg_rank_df = pd.DataFrame({
        'question': questions, 
        'avg_rank': avg_question_ranks, 
        'position': np.argsort(np.argsort(avg_question_ranks))
    }).set_index('question')

    if sort_by_rank:
        sorted_rank_df = avg_rank_df.sort_values(by='position', ascending=False)
    else:
        sorted_rank_df = avg_rank_df.sort_index(ascending=False)
    sorted_questions = sorted_rank_df.index.tolist()

    if show_max_questions is not None:
        n_questions = show_max_questions
        sorted_questions = sorted_questions[-n_questions:]

    y_range = range(n_questions)
    
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (5, n_questions / 2.5)
    plt.figure()

    all_question_ranks_df = pd.DataFrame(all_question_ranks, columns=questions)
    sns.boxplot(data=all_question_ranks_df[sorted_questions], orient='h', color="lightgray")
    
    ytick_labels = ["{} ({:d})".format(question, avg_rank_df['position'][question]+1) 
        for i, question in zip(y_range[::-1], sorted_questions)]
    plt.yticks(y_range, ytick_labels, size=16)
    plt.xticks(size=16)
    if show_question_labels:
        assert question_to_label is not None
        for i in y_range:
            question = sorted_questions[i]
            label = label_for_question(question, question_to_label)
            if not sort_by_rank and avg_rank_df['position'][question] < 10:
                weight = 'bold'
            else:
                weight = 'normal'
            plt.text(n_questions+1, i-0.25, label[:max_label_length], weight = weight, size=16)
    plt.xlim(0, n_questions+1)
    plt.ylim(-1, n_questions)
    subgroup_str = 'all users' if subgroup == 'all' else subgroup
    plt.title("Active question order across simulations, {}".format(subgroup_str), y=1.01, size=20)

    # print('\n'.join(sorted_questions[::-1]))


# # reweight empirical distribution of question vectors to be more uniformly distributed in latent space
# def compute_question_weights(V, matrix_norm='nuc'):
#     r, k = V.shape
#     w = cvx.Variable(k)
#     W = cvx.diag(w)
    
#     objective = cvx.Minimize(cvx.norm(V * W * V.T / k - np.eye(r) / r, matrix_norm))
#     constraints = [w >= 0, cvx.sum(w) == 1]
#     prob = cvx.Problem(objective, constraints)
    
#     assert prob.is_dcp()  # check this is a valid convex program
    
#     optimal_value = prob.solve()
#     print("Optimal value", optimal_value)
    
#     return w.value  # numpy array

# # plot question vectors in latent space with color intensity representing question weight
# def plot_question_weights(V, weights):
#     plt.rcParams["figure.figsize"] = (6, 6)
#     fig, ax = plt.subplots(nrows=1, ncols=1)
#     labels = ["" if np.abs(w_i) < 1e-2 else "q{}".format(i) for i, w_i in enumerate(weights)]
#     colors = plt.cm.Blues(weights)        
#     plot_V_2d(ax, V, labels=labels, colors=colors, title="Question vector weights")
#     plt.show()

# def append_weighted_metrics(sim_results, sim_objects, questions):
#     # compute question weights per simulation
#     weights_df_list = []
#     for uuid, objects in sim_objects.items():
#         V = objects['V']

#         # normalize columns of V to unit sphere in latent space
#         V_col_norm = np.sqrt(np.power(V, 2).sum(axis=0))
#         V = V / V_col_norm[np.newaxis,:]

#         # weights seem to be more dispersed with Frobenius norm instead of nuclear norm
#         weights = compute_question_weights(V, matrix_norm='fro')
#     #     plot_question_weights(V, weights)

#         question_uuid_tuples = [(question, uuid) for question in questions]
#         index = pd.MultiIndex.from_tuples(question_uuid_tuples, names=['question', 'uuid'])
#         weights_df = pd.DataFrame({'question_weight': weights}, index=index)
#         weights_df_list.append(weights_df)

#         # display(weights_df.sort_values(by='question_weight', ascending=False))

#     question_weights_df = pd.concat(weights_df_list)
    
#     # join question weights to sim results
#     # this will exclude entries with columns='all'
#     sim_results_with_weights = sim_results.join(question_weights_df, on=['columns', 'uuid'], how='inner')

#     # compute weighted metrics
#     for metric in Metrics._fields:
#         sim_results_with_weights[metric] *= sim_results_with_weights['question_weight']

#     weighted_results = sim_results_with_weights.groupby(['uuid', 'qnum'])[Metrics._fields].sum()
#     weighted_results['columns'] = 'all-weighted'

#     # join weighted metrics back to metadata so we can append to sim-results
#     # to avoid dupes, use only entries with columns='all'
#     # the overlapping columns from original sim_results will be replaced, by assigning
#     # _unweighted suffix to them and then performing pd.concat with inner join on columns
#     weighted_results_with_metadata = sim_results[sim_results['columns'] == 'all']\
#         .join(weighted_results, on=['uuid', 'qnum'], how='inner', lsuffix='_unweighted')

#     return pd.concat([sim_results, weighted_results_with_metadata], join='inner')



def compute_user_posterior_across_survey(responses, V, question_order, 
    subgroup_list, subgroup_to_mean, subgroup_to_prec, alpha):
    
    R = responses.as_matrix()
    n, k = R.shape
    revealed_responses = np.empty(R.shape)
    revealed_responses[:] = np.nan
    asked_mask = np.repeat(False, k)

    bpmf = BPMF(V, subgroup_list, subgroup_to_mean, subgroup_to_prec, alpha)

    # compute posterior mean sequence
    prior_mean = subgroup_to_mean.get('all')
    # U_init = np.repeat(prior_mean[np.newaxis,:], n, axis=0)  # equivalent to below
    U_init = np.outer(np.ones(n), prior_mean)
    all_U_map = [ U_init ]
    for i, qnum in enumerate(question_order):
        # print("Asking question {:d} in rank {:d}".format(qnum, i))
        asked_mask[qnum] = True
        revealed_responses[:,asked_mask] = R[:,asked_mask]
        completion_result = bpmf.complete(revealed_responses)
        all_U_map.append(completion_result.U)

    # compute posterior variance sequence
    Lambda = subgroup_to_prec.get('all')
    prior_var = np.linalg.inv(Lambda)
    all_posterior_var = [ prior_var ]
    for i in question_order:
        v_i = V[:, i]
        Lambda = Lambda + alpha * np.outer(v_i, v_i)
        posterior_var = np.linalg.inv(Lambda)
        all_posterior_var.append(posterior_var)
        
    return (all_U_map, all_posterior_var)

# plot trajectories of a random subset of users in latent space
# all_U_map is a list of matrices containing MAP estimates of user factors over the survey
def plot_user_mean_trajectories(all_U_map, rank, alpha, user_ids):
    plt.rcParams["figure.figsize"] = (8, 7)
    sns.set_style("ticks")

    for i in range(len(user_ids)):
        all_U_map_i = np.vstack([U_map[i] for U_map in all_U_map])
        x = all_U_map_i[:,0]
        y = all_U_map_i[:,1]
        plt.plot(x, y, label=user_ids[i])

    # plt.legend(title='Respondent', loc=(1.05, 0.05))
    plt.title("Evolution of user factors for active strategy,\nrank={:d}, alpha={:d}".format(rank, alpha))
    plt.axes().set_aspect('equal', 'datalim')

# Get confidence ellipse parameters for 2d normal distribution
# using Syrtis Major's answer in
# https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
def cov_ellipse(cov, q=None, nsig=None, **kwargs):
    """
    Parameters
    ----------
    cov : (2, 2) array
        Covariance matrix.
    q : float, optional
        Confidence level, should be in (0, 1)
    nsig : int, optional
        Confidence level in unit of standard deviations. 
        E.g. 1 stands for 68.3% and 2 stands for 95.4%.

    Returns
    -------
    width, height, rotation :
         The lengths of two axises and the rotation angle in degree
    for the ellipse.
    """

    if q is not None:
        q = np.asarray(q)
    elif nsig is not None:
        q = 2 * norm.cdf(nsig) - 1
    else:
        raise ValueError('One of `q` and `nsig` should be specified.')
    r2 = chi2.ppf(q, 2)

    val, vec = np.linalg.eigh(cov)
    width, height = 2 * np.sqrt(val[:, None] * r2)
    rotation = np.degrees(np.arctan2(*vec[::-1, 0]))

    return width, height, rotation

# plot trajectory of user posterior as confidence ellipse around MAP estimate in latent space
# draw single trajectory, since shape doesn't depend on response values
# all_U_map is a list of matrices containing MAP estimates of user factors over the survey
# all_posterior_var is a list of posterior variance matrices over the survey
def plot_user_posterior_trajectory(all_U_map, all_posterior_var, rank, alpha, user_ids,
    q=None, nsig=None, user_idx=0):

    plt.rcParams["figure.figsize"] = (8, 7)
    sns.set_style("ticks")

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    all_U_map_i = np.vstack([U_map[user_idx] for U_map in all_U_map])

    colors = plt.cm.Greys(np.linspace(0.25, 0.75, len(all_posterior_var)))
    for j in range(len(all_posterior_var)):
        U_map = all_U_map_i[j]
        posterior_var = all_posterior_var[j]
        width, height, rotation = cov_ellipse(posterior_var, q=q, nsig=nsig)
        ellip = Ellipse(xy=U_map, width=width, height=height, angle=rotation, color=colors[j], alpha=0.4)
        ax.add_artist(ellip)

    x = all_U_map_i[:,0]
    y = all_U_map_i[:,1]
    plt.plot(x, y, color='k')

    plt.title("Evolution of one user posterior for active strategy,\nrank={:d}, alpha={:d}".format(rank, alpha))

    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    plt.xlim(xmin-5, xmax+5)
    plt.ylim(ymin-5, ymax+5)
    ax.set_aspect('equal', 'datalim')

def plot_user_factors(sim_objects, responses,
                      rank, optimality_type, v_method, alpha, max_sims=4, n_users=10):
    
    sim_keys = get_active_sim_keys(sim_objects, rank, optimality_type, v_method, alpha)
    sim_key_to_question_order = extract_from_cache(sim_objects, sim_keys, 'question', 'all')
    
    # restrict to random subset of users
    n = responses.shape[0]
    random_users = np.random.choice(n, n_users)
    random_user_ids = responses.index[random_users]

    for sim_key in sim_keys[:max_sims]:
        question_order = sim_key_to_question_order[sim_key]
        sim_object = sim_objects[sim_key]
        subgroup_list = np.repeat('all', n)

        all_U_map, all_posterior_var = compute_user_posterior_across_survey(
            responses.iloc[random_users], 
            sim_object.get('V'),
            question_order,
            subgroup_list[random_users], 
            sim_object.get('subgroup_to_mean'), 
            sim_object.get('subgroup_to_prec'),
            alpha
        )

        plot_user_mean_trajectories(all_U_map, rank, alpha, random_user_ids)
        plot_user_posterior_trajectory(all_U_map, all_posterior_var, rank, alpha, random_user_ids, nsig=2)
