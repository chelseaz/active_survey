
if __name__ == "__main__":
    # tensorflow will block if subprocesses are forked, so spawn them instead. See
    # https://github.com/tensorflow/tensorflow/issues/5448
    import multiprocessing as mp
    mp.set_start_method('spawn')

    import argparse
    import numpy as np
    import os
    import pandas as pd
    import uuid

    from common import ResultsWriter, ObjectsWriter
    from simulate import run

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--selection-methods', type=str, required=True)
    parser.add_argument('--optimality', choices=['A', 'D', 'E'], default='A')
    parser.add_argument('--completion-method', choices=['mean', 'mf', 'soft', 'itersvd', 'bpmf', 'ordlogit'], required=True)
    parser.add_argument('--rank', type=int, default=2)
    parser.add_argument('--v-method', choices=['row-norm', 'col-norm', 'mf'], default='row-norm')
    parser.add_argument('--max-questions', type=int)
    parser.add_argument('--random-seed', type=int, default=1234)  # for training/simulation split only
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--response-filename', type=str)
    parser.add_argument('--results-filename', type=str)
    parser.add_argument('--objects-filename', type=str)
    parser.add_argument('--subgroups', action='store_true')
    parser.add_argument('--subgroup-filename', type=str)
    parser.add_argument('--free-covariates', type=str)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--eval-method', choices=['sparsify', 'lococv', 'kfoldcv'], default='lococv')
    parser.add_argument('--kfoldcv-file', type=str)
    parser.add_argument('--skip-cache', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--save-responses', action='store_true')
    parser.add_argument('--max-n-train', type=int)
    parser.add_argument('--max-n-simul', type=int)
    args = parser.parse_args()

    args.sim_uuid = uuid.uuid4()

    response_filename = 'data/%s.csv' % args.dataset if args.response_filename is None else args.response_filename
    results_filename = 'data/%s-sim-results.csv' % args.dataset if args.results_filename is None else args.results_filename
    objects_filename = 'data/%s-sim-objects.pkl' % args.dataset if args.objects_filename is None else args.objects_filename
    subgroup_filename = 'data/%s-subgroups.csv' % args.dataset if args.subgroup_filename is None else args.subgroup_filename

    results_writer = ResultsWriter(results_filename, replace=args.replace)
    objects_writer = ObjectsWriter(objects_filename, replace=args.replace)

    # np.random.seed(args.random_seed)

    responses = pd.read_csv(response_filename, index_col=0)

    if args.subgroups:
        subgroups = pd.read_csv(subgroup_filename, index_col=0)
    else:
        # create dummy subgroup that everyone belongs to
        subgroups = pd.DataFrame(
            data=dict(bucket=np.repeat('all', responses.shape[0])), 
            index=responses.index
        )

    run(responses, subgroups, results_writer, objects_writer, args)

    # tf shutdown hook?
