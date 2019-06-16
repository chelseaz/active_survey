#!/bin/bash

echo "Using dataset" $DATASET
echo "Using response filename" $RESPONSE_FILENAME
echo "Using subgroup filename" $SUBGROUP_FILENAME
echo "Using $REPS replications"

set -ex

OPTIM=A  #D E
VMETHOD=row-norm  #mf col-norm

if [ -z $FREE_COVARIATES ]; then
    FREE_COVARIATES_ARG=""
else
    FREE_COVARIATES_ARG="--free-covariates $FREE_COVARIATES"
fi

if [ -z $SUBGROUP_FILENAME ]; then
    SUBGROUPS_ARG=""
else
    SUBGROUPS_ARG="--subgroups --subgroup-filename $SUBGROUP_FILENAME"
fi

if [ -z $MAX_TRAIN ]; then
    MAX_TRAIN_ARG=""
else
    MAX_TRAIN_ARG="--max-n-train $MAX_TRAIN"
    DATASET="${DATASET}_t${MAX_TRAIN}"
fi

if [ -z $MAX_SIMUL ]; then
    MAX_SIMUL_ARG=""
else
    MAX_SIMUL_ARG="--max-n-simul $MAX_SIMUL"
    DATASET="${DATASET}_s${MAX_SIMUL}"
fi

for REP in `seq 1 $REPS`
do
    if [ "$MODE" = "eval" ]; then
        MODE_ARG="--skip-cache"
    elif [ "$MODE" = "order" ]; then
        DATESEC=`date +%s`
        MODE_ARG="--skip-eval --random-seed $(($DATESEC+$REP))"
    else
        MODE_ARG=""
    fi

    for RANK in 4 #2 4 #8
    do
        for ALPHA in 1 #10 100
        do
            echo "Rank $RANK, alpha $ALPHA"
            echo "Replication $REP of $REPS"
            python active_sampling.py \
                --dataset ${DATASET} \
                --selection-methods $SELECTION_METHODS \
                --optimality $OPTIM \
                --completion-method $COMPLETER \
                --rank $RANK \
                --v-method $VMETHOD \
                --response-filename $RESPONSE_FILENAME \
                --alpha $ALPHA \
                --eval-method $EVAL_METHOD \
                $MAX_TRAIN_ARG \
                $MAX_SIMUL_ARG \
                $FREE_COVARIATES_ARG \
                $SUBGROUPS_ARG \
                $MODE_ARG #\
                # --kfoldcv-file data/ec2/folds.json
        done
    done
done
