import numpy as np
import os
import pickle

from collections import namedtuple
from fancyimpute import SimpleFill, IterativeSVD, MatrixFactorization, SoftImpute
from functools import wraps
from time import time

CompletionResult = namedtuple('CompletionResult', ['X_filled', 'U', 'V', 'S', 'rank'])

OptCriteria = namedtuple('OptCriteria', ['trace', 'logdet', 'max_eigval'])


def get_completer(method, rank):
    if method == 'mean':
        return SimpleFill()
    elif method == 'mf':
        # Note the fancyimpute docs say this imposes an L1 penalty
        # on U and an L2 penalty on V. However, the code suggests 
        # this uses an L2 penalty for both.
        return MatrixFactorization(rank=rank, use_bias=False)
    elif method == 'soft':
        return SoftImpute(max_rank=rank, init_fill_method="mean", verbose=False)
    elif method == 'itersvd':
        return IterativeSVD(rank=rank, init_fill_method="mean")
    else:
        raise NotImplementedError


# Lambda is the precision matrix
# return measures of the variance, Lambda^{-1}
def compute_opt_criteria(Lambda):
    eigvals_Lambda = np.linalg.eigvals(Lambda)
    # When there are not enough previous questions for full rank,
    # some eigenvalues will be 0. Ensure numerical stability here.
    # Originally chose threshold 1e-8, but sometimes an eigenvalue
    # that should've been 0 exceeded this threshold.
    # TODO: handle in a safer way.
    eigvals_Lambda[eigvals_Lambda < 1e-6] = 1e-6
    eigvals_Var = 1.0 / eigvals_Lambda
    return OptCriteria(
        trace = np.sum(eigvals_Var),
        logdet = np.sum(np.log(eigvals_Var)),
        max_eigval = np.max(eigvals_Var)
    )

# these used to be lambdas but python couldn't pickle them
def trace_comparator(o1, o2):
    return o1.trace - o2.trace

def logdet_comparator(o1, o2):
    return o1.logdet - o2.logdet

def max_eigval_comparator(o1, o2):
    return o1.max_eigval - o2.max_eigval

opt_criteria_comparators = dict(
    A = trace_comparator,
    D = logdet_comparator,
    E = max_eigval_comparator
)

def cmp_criteria_for(optimality_type):
    return opt_criteria_comparators[optimality_type]


def serialize(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def deserialize(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class ResultsWriter(object):
    def __init__(self, results_filename, replace=False):
        self.results_filename = results_filename
        if replace and os.path.exists(self.results_filename):
            # remove existing file
            os.remove(self.results_filename)

    def save_results(self, results_df):
        if not os.path.exists(self.results_filename):
            # write results to new file
            results_df.to_csv(self.results_filename, mode='w', header=True, index=False,
                float_format='%.3f')
        else:
            # append results to file
            results_df.to_csv(self.results_filename, mode='a', header=False, index=False,
                float_format='%.3f')


class ObjectsWriter(object):
    def __init__(self, objects_filename, replace=False):
        self.objects_filename = objects_filename
        if replace and os.path.exists(self.objects_filename):
            # remove existing file
            os.remove(self.objects_filename)

    def save_objects(self, sim_key, sim_objects):
        # unfortunately de- and re-serializes all objects with every call
        # TODO: just append or save separate files
        if not os.path.exists(self.objects_filename):
            objects = dict()
        else:
            objects = deserialize(self.objects_filename)

        objects[sim_key] = sim_objects
        serialize(objects, self.objects_filename)


# Append simulation results from first file to end of second file.
def transfer_results(from_filename, to_filename):
    header = True
    with open(from_filename, 'r') as from_file:
        with open(to_filename, 'a') as to_file:
            for line in from_file:
                if not header:
                    to_file.write(line)
                header = False


# Copy simulation objects from one file to another, 
# preserving existing objects in destination file.
def transfer_objects(from_filename, to_filename):
    transfer_objects = deserialize(from_filename)
    if not os.path.exists(to_filename):
        all_objects = dict()
    else:
        all_objects = deserialize(to_filename)

    all_objects.update(transfer_objects)
    serialize(all_objects, to_filename)


# From https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('Function %r took %.2f sec' % (f.__name__, te-ts))
        return result
    return wrap
