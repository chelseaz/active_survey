#!/bin/bash

# get many replications to check validity of error bars
RESPONSE_FILENAME=data/cces/cces16_cs.csv
REPS=10

for REP in `seq 31 50`
do
    DATASET=cces16_$REP
    echo "Saving dataset $DATASET"

    source simulate-all.sh
done