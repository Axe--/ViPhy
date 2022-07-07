"""
Evaluate Models on VisCS dataset.
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from typing import List, Dict, Union, Any
from transformers import AutoTokenizer
from collections import Counter
from sklearn.metrics import f1_score
from utils import read_json
from color.constants import COLOR_SET
from models import (ZeroShotLM, MaskedLM, UnifiedQA, ViLT)


def _eval_colors(dataset, model, top_k):
    is_correct = []
    preds = []

    for d in tqdm(dataset):
        # sample
        inp = d['prompt']
        true = d['labels']
        # inference
        if top_k:
            pred = model(inp, top_k)
            # clean
            pred = [p.lower().strip() for p in pred]
            # overlaps with true set
            cond = len(set(true) & set(pred)) > 0

            preds += pred
        else:
            pred = model(inp)
            # clean
            pred = pred.lower().strip()
            # belongs to true set
            cond = pred in true

            preds += [pred]

        # append
        is_correct.append(cond)

    # Metrics
    acc = np.mean(is_correct)

    print(f'\n---- {model.name} ----\n')
    print(f'Top-K: {top_k}')

    print('Accuracy** : {:.4f}'.format(acc*100))
    # print('F1-score: {:.4f}'.format(f1))
    # print('Acc @K={}: {:.4f}'.format(k, acc_top_k*100))

    print('\n', Counter(preds), '\n')


def _color_prompts(obj_colors: Dict[str, Dict[str, float]],
                   task_type: str, model) -> List[Dict]:
    """
    Given object name & associated typical colors,
    generates textual prompts for model evaluation.

    :return: prompts -- [num_objs]
    """
    # Masked LM
    if task_type == 'fill-mask':
        template = "{} is of {} color".format('{}', model.mask)

    # Text-to-Text
    elif task_type == 'text-generation':
        template = "the color of {} is"

    # Question-Answer
    else:
        template = "what is the color of {}? \n"
        # Color choices
        idx = 'a'
        for color in COLOR_SET:
            template += f'({idx}) {color} '

            idx = _increment_char(idx)

    dataset = []

    for obj, colors in obj_colors.items():
        # Model I/O
        prompt = template.format(obj)
        labels = list(colors.keys())

        sample = dict(object=obj, prompt=prompt, labels=labels)

        dataset.append(sample)

    return dataset


def _eval_spatial(dataset, model, top_k):
    is_correct = []
    preds = []

    for d in tqdm(dataset):
        # sample
        inp = d['prompt']
        true = d['labels']
        # inference
        if top_k:
            pred = model(inp, top_k)
            # clean
            pred = [p.lower().strip() for p in pred]
            # overlaps with true set
            cond = len(set(true) & set(pred)) > 0

            preds += pred
        else:
            pred = model(inp)
            # clean
            pred = pred.lower().strip()
            # belongs to true set
            cond = pred in true

            preds += [pred]

        # append
        is_correct.append(cond)

    # Metrics
    acc = np.mean(is_correct)

    print(f'\n---- {model.name} ----\n')
    print(f'Top-K: {top_k}')

    print('Accuracy** : {:.4f}'.format(acc * 100))
    # print('F1-score: {:.4f}'.format(f1))
    # print('Acc @K={}: {:.4f}'.format(k, acc_top_k*100))

    print('\n', Counter(preds), '\n')


def _spatial_prompts(spatial_rels: List[Dict[str, Any]],
                     task_type: str, model) -> List[Dict]:
    """
    Given object name & associated typical colors,
    generates textual prompts for model evaluation.

    :return: prompts -- [num_objs]
    """
    # Masked LM
    if task_type == 'fill-mask':
        template = f"in a {'{}'}, the {'{}'} is located {model.mask} the {'{}'}"

    # Text-to-Text
    elif task_type == 'text-generation':
        template = "?"

    # Question-Answer
    else:
        template = "in a {}, where is {} located in comparison to {}? \n"
        # Color choices
        idx = 'a'
        for choice in ['above', 'below', 'same level']:
            template += f'({idx}) {choice} '

            idx = _increment_char(idx)

    dataset = []
    for rel in spatial_rels:
        # Model I/O
        prompt = template.format(rel['scene'], rel['o1'], rel['o2'])

        if task_type == 'fill-mask' and '=' in rel['typical']:
            continue

        labels = rel['typical'].replace('>', 'above')\
                               .replace('<', 'below')\
                               .replace('=', 'same level')
        labels = labels.split(',')

        sample = dict(o1=rel['o1'], o2=rel['o2'],
                      prompt=prompt, labels=labels)

        dataset.append(sample)

    return dataset


def _increment_char(c: str):
    return chr(ord(c) + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Models")

    parser.add_argument("--task",   type=str)
    parser.add_argument("--model",  type=str)
    parser.add_argument("--eval",   type=str)
    parser.add_argument("--gpu",    type=int)
    parser.add_argument("--top_k",  type=int)

    args = parser.parse_args()

    # Args
    args = edict()
    args.task = 'QA'            # 'QA'
    args.model = 'allenai/unifiedqa-t5-base'    # 'allenai/unifiedqa-t5-large'
    args.eval = 'spatial'
    args.top_k = None
    args.gpu = 1

    # Device
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    # Model
    if 'qa-t5' in args.model:
        model = UnifiedQA(name=args.model, device=device)

    elif 'gpt-neo' in args.model:
        ...

    elif 'ofa' in args.model:
        ...

    else:
        if args.task == 'fill-mask':
            model = MaskedLM(name=args.model, device=device)
        # else:
        #     model = LanguageModel(task=args.task, name=args.model, device=device)

    # Read --> Dataset --> Evaluate
    if args.eval == 'color':
        data = read_json('./results/colors.json')

        data = _color_prompts(obj_colors=data,
                              task_type=args.task,
                              model=model)

        _eval_colors(data, model, args.top_k)

    elif args.eval == 'spatial':
        data = pd.read_csv('./results/spatial.csv')
        data = data.to_dict(orient='records')

        data = _spatial_prompts(spatial_rels=data,
                                task_type=args.task,
                                model=model)

        _eval_spatial(data, model, args.top_k)

    else:
        raise NotImplemented()
