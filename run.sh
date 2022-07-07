#!/bin/bash

# Color: *BERT
#DIM=color
#RUN=bert_base
#MODEL=bert-base-uncased
#GPU=0

# Spatial: *BERT
DIM=spatial
GPU=0

RUN=bert_base
MODEL=bert-base-uncased

#RUN=bert_large
#MODEL=bert-large-uncased
#
#RUN=roberta_base
#MODEL=roberta-base
#
#RUN=roberta_large
#MODEL=roberta-large

python3 main.py --mode train --expt_dir log --expt_name $DIM --run_name $RUN \
--model $MODEL --data ./dataset/$DIM --batch 8 --gpu $GPU --num_worker 1 --save_all F

python3 main.py --mode eval --expt_dir log --expt_name $DIM --run_name $RUN \
--model $MODEL --data ./dataset/$DIM --batch 8 --gpu $GPU

