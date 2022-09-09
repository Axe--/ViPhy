#!/bin/bash

eval() {
  python3 zero_shot.py --task "$1" --model "$2" --eval "$3" --gpu "$4" --top_k "$5"
}

# Args
T='fill-mask'
D='spatial'
k=1
g=0


M="bert-large-uncased"
eval $T $M $D $g $k

M="roberta-large"
eval $T $M $D $g $k

M="uclanlp/visualbert-vqa-coco-pre"
eval $T $M $D $g $k

M="dandelin/vilt-b32-mlm"
eval $T $M $D $g $k

M="facebook/flava-full"
eval $T $M $D $g $k


T="QA"; M="allenai/unifiedqa-t5-large"
eval $T $M $D $g $k

# GPT-2
#T="text-generation"; M="gpt2";