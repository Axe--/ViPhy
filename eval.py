"""
Evaluate Models on VisCS dataset.
- BERT  (mask)
- UniQA (txt)
- OFA   (txt)
"""
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from typing import List, Dict, Union
from transformers import AutoTokenizer
from sklearn.metrics import (f1_score,
                             accuracy_score,
                             confusion_matrix)
from utils import read_json
from color.constants import COLOR_SET
from models import LanguageModel, UnifiedQA


def _eval_spatial():
    relations = pd.read_json('./spatial.json')

    predictions = []

    for rel in relations:
        loc, o1, o2 = rel[''], rel[''], rel['']

        txt = f'In {loc}, {o1} is located [MASK] {o2}'

        pred = None
        predictions.append(pred)


def _eval_colors(dataset, _model):
    is_correct = []
    preds = []

    for d in tqdm(dataset):
        # sample
        inp = d['prompt']
        true_set = d['labels']
        # inference
        pred = _model(inp)
        # clean
        pred = pred.lower().strip()

        is_correct.append(pred in true_set)
        preds.append(pred)

    # Metrics
    acc = np.mean(is_correct)

    print('Accuracy** : {:.4f}'.format(acc*100))
    print(set(preds))


def _color2prompts(obj_colors: Dict[str, Dict[str, float]],
                   task_type: str,
                   model_name: str) -> List[Dict]:
    """
    Given object name & associated typical colors,
    generates textual prompts for model evaluation.

    :return: prompts -- [num_objs]
    """
    def _increment_char(c: str):
        return chr(ord(c) + 1)

    # Masked LM
    if task_type == 'fill-mask':
        tok = AutoTokenizer.from_pretrained(model_name)

        template = "{} is of {} color".format('{}', tok.mask_token)

    # Text-to-Text
    elif task_type == 'text-generation':
        template = "the color of {} is"

    # Question-Answer
    else:
        template = "what is the color of {}? \n"
        # Color choices
        idx = 'a'
        for color in COLOR_SET:
            template += f'({idx}) {color}'

            idx = _increment_char(idx)

    dataset = []

    for obj, colors in obj_colors.items():
        # Model I/O
        prompt = template.format(obj)
        labels = list(colors.keys())

        sample = dict(object=obj, prompt=prompt, labels=labels)

        dataset.append(sample)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Region Object Subtypes")

    parser.add_argument("--task",   type=str, required=True)
    parser.add_argument("--model",  type=str, required=True)
    parser.add_argument("--gpu",    type=int, required=True)
    parser.add_argument("--color",  type=str, default='./results/colors.json')

    args = parser.parse_args()

    # Args
    # args = edict()
    # args.task = 'QA'    # 'fill-mask'
    # args.model = 'allenai/unifiedqa-t5-large'   # 'bert-base-uncased'
    # args.color = './results/colors.json'
    # args.gpu = 1

    # Device
    device = 'cuda:{}'.format(args.gpu) \
             if args.gpu >= 0 else 'cpu'

    # Typical Colors
    jsn = read_json(args.color)

    # Dataset
    data = _color2prompts(obj_colors=jsn,
                          task_type=args.task,
                          model_name=args.model)
    # Model
    if 'qa' in args.model:
        model = UnifiedQA(name=args.model,
                          device=device)
    else:
        model = LanguageModel(task=args.task,
                              name=args.model,
                              device=device)

    # Evaluate
    _eval_colors(data, model)
