#!/bin/bash

MODE=eval
COMPLETER=bpmf  #soft
SELECTION_METHODS=epsilon_greedy,active,random,sequential

for EVAL_METHOD in sparsify  #lococv kfoldcv
do
    if [[ "$EVAL_METHOD" == "sparsify" ]]
    then
        REPS=10
    else
        REPS=1
    fi

    DATASET=cces18
    RESPONSE_FILENAME=data/cces/cces18_cs.csv
    source simulate-all.sh

    DATASET=cces16
    RESPONSE_FILENAME=data/cces/cces16_cs.csv
    source simulate-all.sh

    DATASET=cces12
    RESPONSE_FILENAME=data/cces/cces12_cs.csv
    source simulate-all.sh

    DATASET=cces16_free_covariates
    RESPONSE_FILENAME=data/cces/cces16_full_cs.csv
    FREE_COVARIATES=ideo5,pid7,pew_religimp,newsint,pid3_Democrat,pid3_Independent,pid3_Republican,gender_Female,gender_Male,educ_2-year,educ_4-year,educ_High_school_graduate,educ_Post-grad,educ_Some_college,race_Asian,race_Black,race_Hispanic,race_Mixed,race_White,child18_No,child18_Yes,ownhome_Own,ownhome_Rent
    source simulate-all.sh
done

COMPLETER=ordlogit
MAX_TRAIN=1000
MAX_SIMUL=1000
for EVAL_METHOD in sparsify
    DATASET=cces16
    RESPONSE_FILENAME=data/cces/cces16_ord.csv
    REPS=1
    source simulate-all.sh
done