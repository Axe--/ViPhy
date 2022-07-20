#!/bin/bash

eval() {
  python3 eval_zs.py --task "$1" --model "$2" --eval "$3" --gpu "$4" --top_k "$5"
}

# Args
T='fill-mask'
E='color'
g=0


# *BERT*
M="bert-base-uncased"
k=1; eval $T $M $E $g $k
k=3; eval $T $M $E $g $k

M="bert-large-uncased"
k=1; eval $T $M $E $g $k
k=3; eval $T $M $E $g $k


# *RoBERTa*
M="roberta-base"
k=1; eval $T $M $E $g $k
k=3; eval $T $M $E $g $k

M="roberta-large"
k=1; eval $T $M $E $g $k
k=3; eval $T $M $E $g $k


# GPT-2
#T="text-generation"; M="gpt2";

# Unified-QA
#T="QA"; M="allenai/unifiedqa-t5-base"
