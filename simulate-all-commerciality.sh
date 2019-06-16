#!/bin/bash

MODE=order  #eval
COMPLETER=bpmf  #ordlogit bpmf soft
SELECTION_METHODS=epsilon_greedy,active,random,sequential

for EVAL_METHOD in sparsify  #lococv kfoldcv
do
    if [[ "$EVAL_METHOD" == "sparsify" ]]
    then
        REPS=10
    else
        REPS=1
    fi

    DATASET=commerciality
    RESPONSE_FILENAME=data/commerciality/commerciality_cs.csv 
    source simulate-all.sh

    # DATASET=commerciality_country_tenure4
    # RESPONSE_FILENAME=data/commerciality/commerciality_cs.csv 
    # SUBGROUP_FILENAME=data/commerciality/user_to_country_tenure4.csv
    # source simulate-all.sh
done