#!/bin/bash

run() {
  python3 few_shot.py --model "$1" --data "$2" --k "$3"
}

# ------------------------
DATA="./dataset/fs_size"
# ------------------------

MODEL=bert-base-uncased
K=20;  run $MODEL $DATA $K &
K=40;  run $MODEL $DATA $K &
K=60;  run $MODEL $DATA $K &
K=80;  run $MODEL $DATA $K &
K=100; run $MODEL $DATA $K


MODEL=roberta-base
K=20;  run $MODEL $DATA $K &
K=40;  run $MODEL $DATA $K &
K=60;  run $MODEL $DATA $K &
K=80;  run $MODEL $DATA $K &
K=100; run $MODEL $DATA $K


MODEL=microsoft/deberta-base
K=20;  run $MODEL $DATA $K &
K=40;  run $MODEL $DATA $K &
K=60;  run $MODEL $DATA $K &
K=80;  run $MODEL $DATA $K &
K=100; run $MODEL $DATA $K


MODEL=dandelin/vilt-b32-mlm
K=20;  run $MODEL $DATA $K &
K=40;  run $MODEL $DATA $K &
K=60;  run $MODEL $DATA $K &
K=80;  run $MODEL $DATA $K &
K=100; run $MODEL $DATA $K


MODEL=openai/clip-vit-base-patch32
K=20;  run $MODEL $DATA $K &
K=40;  run $MODEL $DATA $K &
K=60;  run $MODEL $DATA $K &
K=80;  run $MODEL $DATA $K &
K=100; run $MODEL $DATA $K


MODEL=facebook/flava-full
K=20;  run $MODEL $DATA $K &
K=40;  run $MODEL $DATA $K &
K=60;  run $MODEL $DATA $K &
K=80;  run $MODEL $DATA $K &
K=100; run $MODEL $DATA $K


MODEL=CapBERT
K=20;  run $MODEL $DATA $K &
K=40;  run $MODEL $DATA $K &
K=60;  run $MODEL $DATA $K &
K=80;  run $MODEL $DATA $K &
K=100; run $MODEL $DATA $K
