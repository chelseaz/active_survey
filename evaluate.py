import json
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import KFold

from collections import namedtuple
from sparsify import clobber, UserSparsity

Metrics = namedtuple('Metrics', ['mse', 'mae', 'sdse', 'sdae', 'pws', 'bias', 'n'])

def binary_sign(array):
    three_way_sign = np.sign(array)
    three_way_sign[three_way_sign == 0] = 1
    return three_way_sign

def mean_and_std(array):
    return array.mean(), array.std()

# truth and preds are 1d arrays
def compute_metrics(truth, preds):
    mse, sdse = mean_and_std(np.power(truth - preds, 2))
    mae, sdae = mean_and_std(np.abs(truth - preds))
    return Metrics(
        mse = mse, mae = mae, sdse = sdse, sdae = sdae,
        pws = (binary_sign(truth) != binary_sign(preds)).mean(),
        # pws = proportion with wrong prediction sign
        bias = (preds - truth).mean(),
        n = truth.size
    )


# A container for one of several validation sets.
class CVFold(object):
    def __init__(self, name, R, validation_mask):
        self.name = name
        # validation_mask: matrix of same shape as R, containing which entries to NA out
        self.validation_mask = validation_mask
        # R_train: a copy of R with entries in the validation set NAed out
        self.R_train = clobber(R, validation_mask)

        # validation_cols: length-k boolean vector. True columns are excluded from 
        # being asked by the simulated survey.
        # A column is True only if the entire column is True in validation_mask.
        self.validation_cols = np.all(self.validation_mask, axis=0)

        # evaluation_cols: length-k boolean vector. True columns will have metrics evaluated.
        # A column is True if it has any True entries in validation_mask.
        self.evaluation_cols = np.any(self.validation_mask, axis=0)


# Generator that yields folds by punching random holes in each row
# eval_blacklist is None or a boolean array of length k
def evaluation_folds_sparsify(R, n_reps=1, sparsity_level=0.2, eval_blacklist=None):
    sparsifier = UserSparsity(sparsity_level)
    for i in range(n_reps):
        clobber_mask = sparsifier.get_clobber_mask(R)
        if eval_blacklist is not None:
            clobber_mask[:, eval_blacklist] = False

        yield CVFold(name=str(i), R=R, validation_mask=clobber_mask)


# Generator that yields folds using leave-one-column-out cross-validation
# eval_blacklist is None or a boolean array of length k
def evaluation_folds_lococv(R, questions, eval_blacklist=None):
    k = R.shape[1]
    for i in range(k):
        if eval_blacklist is not None and eval_blacklist[i]:
            print("Column {} blacklisted from evaluation, skipping".format(questions[i]))
            continue

        # leave out column i
        holdout_mask = np.zeros(R.shape)
        holdout_mask[:,i] = 1
        holdout_mask = holdout_mask.astype(bool)
        yield CVFold(
            name="-"+questions[i], R=R,
            validation_mask=holdout_mask
        )


# Generator that yields folds using k-fold cross-validation, choosing folds randomly
# eval_blacklist is None or a boolean array of length k
def evaluation_folds_kfoldcv(R, k=5, eval_blacklist=None):
    if eval_blacklist is None:
        eval_blacklist = np.repeat(False, R.shape[1])
    eval_q_idx, = np.where(~eval_blacklist)

    kf = KFold(n_splits=k, shuffle=True)
    for _, holdout_q_idx in kf.split(eval_q_idx):
        # leave out all columns in holdout_q_idx
        holdout_mask = np.zeros(R.shape)
        holdout_mask[:,holdout_q_idx] = 1
        holdout_mask = holdout_mask.astype(bool)
        yield CVFold(
            name=','.join(holdout_q_idx.astype(str).tolist()),
            R=R,
            validation_mask=holdout_mask
        )


# Generator that yields folds saved in `filename` as JSON
# Object is a list of folds. Each fold is a list of question names.
def evaluation_folds_from_file(R, questions, filename):
    with open(filename, 'r') as f:
        folds = json.loads(f.readline())

    for fold in folds:
        # leave out all questions in fold
        holdout_mask_vec = [q in fold for q in questions]
        holdout_mask = np.zeros(R.shape)
        holdout_mask[:,holdout_mask_vec] = 1
        holdout_mask = holdout_mask.astype(bool)
        yield CVFold(
            name=','.join(fold),
            R=R,
            validation_mask=holdout_mask
        )


def evaluation_loop(R, survey_iterator, results_writer, args, questions, eval_blacklist=None):
    notna_mask = ~np.isnan(R)

    if args.eval_method == 'sparsify':
        fold_generator = evaluation_folds_sparsify(R, eval_blacklist=eval_blacklist)
    elif args.eval_method == 'lococv':
        fold_generator = evaluation_folds_lococv(R, questions, eval_blacklist=eval_blacklist)
    elif args.eval_method == 'kfoldcv' and args.kfoldcv_file is None:
        fold_generator = evaluation_folds_kfoldcv(R, eval_blacklist=eval_blacklist)
    elif args.eval_method == 'kfoldcv' and args.kfoldcv_file is not None:
        fold_generator = evaluation_folds_from_file(R, questions, args.kfoldcv_file)
    else:
        raise ValueError("Unrecognized evaluation method %s" % method)
    
    # eventually, compute this in parallel
    for fold in fold_generator:
        print("Evaluating on fold %s" % fold.name)

        # exclude originally missing responses from validation set
        validation_mask = np.logical_and(fold.validation_mask, notna_mask)
        print("Validation set contains %d values" % validation_mask.sum())

        sim_results = []
        evaluation_col_idx, = np.where(fold.evaluation_cols)

        # get survey state at specified iterations
        for iteration in survey_iterator(fold):
            complete_responses = iteration.complete_responses

            if args.eval_method == 'sparsify':
                # don't save matrix results for LOCOCV, since they will simply
                # duplicate column results
                matrix_results = results_row_for(
                    columns='all',
                    qnum=iteration.qnum,
                    args=args)
                metrics = compute_metrics(
                    truth = R[validation_mask],
                    preds = complete_responses[validation_mask]
                )
                matrix_results.update(metrics._asdict())
                sim_results.append(matrix_results)

            for i in evaluation_col_idx:
                col_validation_mask = validation_mask[:, i]
                column_metrics = compute_metrics(
                    truth = R[col_validation_mask, i],
                    preds = complete_responses[col_validation_mask, i]
                )
                column_results = results_row_for(
                    columns=questions[i],
                    qnum=iteration.qnum,
                    args=args)
                column_results.update(column_metrics._asdict())
                sim_results.append(column_results)

        # Save results incrementally
        results_writer.save_results(pd.DataFrame(sim_results))



def results_row_for(columns, qnum, args):
    return {
        'eval_method': args.eval_method,
        'selector': args.selection_method,
        'optimality': args.optimality,
        'completer': args.completion_method,
        'rank': args.rank,
        'v_method': args.v_method,
        'alpha': args.alpha,
        'uuid': args.sim_uuid,
        'columns': columns,
        'qnum': qnum
    }
