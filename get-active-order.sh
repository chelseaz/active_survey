#!/bin/bash

set -ex

# use cces16_base to prevent clashing with simulations involving cces16
DATASETS="cces16_base
cces16_with_pol_ids
cces16_with_demos
cces16_full"
REPS=10
RANK=4
ALPHA=1
OPTIM=A
VMETHOD=row-norm
COMPLETER=bpmf
EVAL_METHOD=sparsify

for DATASET in $DATASETS
do
    for REP in `seq 1 $REPS`
    do
        RESPONSE_FILENAME="data/cces/${DATASET}_cs.csv"
        python active_sampling.py \
            --dataset $DATASET \
            --selection-methods active \
            --optimality $OPTIM \
            --completion-method $COMPLETER \
            --rank $RANK \
            --v-method $VMETHOD \
            --response-filename $RESPONSE_FILENAME \
            --alpha $ALPHA \
            --eval-method $EVAL_METHOD \
            --skip-eval \
            --max-n-simul 10  # more efficient, shouldn't affect active order
    done
done