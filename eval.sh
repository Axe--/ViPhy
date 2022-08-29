#!/bin/bash

run() {
  python3 main.py --mode "$1" --expt_dir "$2" --expt_name "$3" --run "$4" --model "$5" --data "$6" --gpu "$7"
}

# ------------------------
LOG="log"
MODE="eval"
GPU=1
DIM="color"
DATA="./dataset/$DIM"
# ------------------------


RUN=bert_base; MODEL=bert-base-uncased; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=bert_large; MODEL=bert-large-uncased; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=roberta_base; MODEL=roberta-base; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=roberta_large; MODEL=roberta-large; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=clip; MODEL=openai/clip-vit-base-patch32; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=flava; MODEL=facebook/flava-full; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=vilt; MODEL=dandelin/vilt-b32-mlm; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=visbert; MODEL=uclanlp/visualbert-vqa-coco-pre; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=deberta_base; MODEL=microsoft/deberta-base; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=deberta_xxl; MODEL=microsoft/deberta-v2-xxlarge; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=uniqa_base; MODEL=allenai/unifiedqa-t5-base; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU

RUN=uniqa_large; MODEL=allenai/unifiedqa-t5-large; run $MODE $LOG $DIM $RUN $MODEL $DATA $GPU
