"""
Zero-shot Evaluation on ViPhy datasets.
"""
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from typing import List, Dict, Union, Any
from collections import Counter
from sklearn.metrics import f1_score
from utils import read_json, read_csv
from color.constants import COLOR_SET
from models import (ZeroShotLM, MaskedLM, UnifiedQA)


def _color_prompts(obj_colors: Dict[str, Dict[str, float]],
                   _type: str, model) -> List[Dict]:
    """
    Given object name & associated typical colors,
    generates textual prompts for model evaluation.

    :return: prompts -- [num_objs]
    """
    # Masked LM
    if _type == 'fill-mask':
        mask = model.tokenizer.mask_token

        template = f"{'{}'} is of {mask} color"

    # Text-to-Text
    elif _type == 'text-generation':
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


def _spatial_prompts(spatial_rels: List[Dict[str, Any]],
                     _type: str, model) -> List[Dict]:
    """
    Given object name & associated typical colors,
    generates textual prompts for model evaluation.

    :return: prompts -- [num_objs]
    """
    # Masked LM
    if _type == 'fill-mask':
        mask = model.tokenizer.mask_token

        template = f"in a {'{}'}, the {'{}'} is located {mask} the {'{}'}"

    # Text-to-Text
    elif _type == 'text-generation':
        template = "?"

    # Question-Answer
    else:
        template = "in a {}, where is {} located in comparison to {}? \n"
        # Color choices
        idx = 'a'
        for choice in ['above', 'below']:   # 'similar level'
            template += f'({idx}) {choice} '

            idx = _increment_char(idx)

    dataset = []
    for rel in spatial_rels:
        # Model I/O
        prompt = template.format(rel['scene'], rel['o1'], rel['o2'])

        if '=' in rel['typical']:
            continue

        labels = rel['typical'].replace('>', 'above')\
                               .replace('<', 'below')\
                               .replace('=', 'similar level')
        labels = labels.split(',')

        sample = dict(o1=rel['o1'], o2=rel['o2'],
                      prompt=prompt, labels=labels)

        dataset.append(sample)

    return dataset


def _size_prompts(size_rels: List[Dict[str, Any]],
                  _type: str, model) -> List[Dict]:
    """
    Given object name & associated typical colors,
    generates textual prompts for model evaluation.

    :return: prompts -- [num_objs]
    """
    # Masked LM
    if _type == 'fill-mask':
        mask = model.tokenizer.mask_token

        template = f"{'{}'} is {mask} than {'{}'} in size"

    # Text-to-Text
    elif _type == 'text-generation':
        template = "?"

    # Question-Answer
    else:
        template = "what is the size of {} in comparison to {}? \n"
        # Color choices
        idx = 'a'
        for choice in ['smaller', 'larger']:
            template += f'({idx}) {choice} '

            idx = _increment_char(idx)

    dataset = []
    for rel in size_rels:
        # Model I/O
        prompt = template.format(rel['o1'], rel['o2'])

        labels = ['bigger', 'larger'] if rel['typical'] == '>' else ['smaller']

        sample = dict(o1=rel['o1'], o2=rel['o2'],
                      prompt=prompt, labels=labels)

        dataset.append(sample)

    return dataset


def _increment_char(c: str):
    return chr(ord(c) + 1)


def _eval(dataset, model, top_k):
    is_correct = []
    preds = []

    for d in tqdm(dataset):
        # sample
        inp = d['prompt']
        true = d['labels']
        # inference
        pred = model.predict(inp, top_k)
        # clean
        if type(pred) == str:
            pred = [pred]
        pred = [p.lower().strip() for p in pred]
        # overlap with true set
        cond = len(set(true) & set(pred)) > 0

        preds += pred

        # append
        is_correct.append(cond)

    # Metrics
    acc = np.mean(is_correct)

    print(f'\n---- {model.name} ----\n')
    print(f'Top-K: {top_k}')
    print('Accuracy** : {:.4f}'.format(acc * 100))
    print('\n', Counter(preds), '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Models")

    parser.add_argument("--task",   type=str)
    parser.add_argument("--model",  type=str)
    parser.add_argument("--eval",   type=str)
    parser.add_argument("--gpu",    type=int)
    parser.add_argument("--top_k",  type=int)

    args = parser.parse_args()

    # Args
    # args = edict()
    # args.task = 'QA'    # 'fill-mask'
    # args.model = 'allenai/unifiedqa-t5-large'
    # args.eval = 'size'
    # args.top_k = 1
    # args.gpu = 1

    # Device
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    # Model
    if 'qa-t5' in args.model:
        _model = UnifiedQA(name=args.model, device=device)

    else:
        if args.task == 'fill-mask':
            _model = MaskedLM(name=args.model, num_classes=0, zs=True, device=device)
        else:
            _model = ZeroShotLM(task=args.task, name=args.model, device=device)

    # Read --> Dataset --> Evaluate
    if args.eval == 'color':
        data = read_json('./dataset/color/test.json')

        data = _color_prompts(obj_colors=data,
                              _type=args.task,
                              model=_model)

    elif args.eval == 'spatial':
        data = read_csv('./dataset/spatial/test.csv')

        data = _spatial_prompts(spatial_rels=data,
                                _type=args.task,
                                model=_model)
    else:
        data = read_csv('./dataset/size/test.csv')

        data = _size_prompts(size_rels=data,
                             _type=args.task,
                             model=_model)

    _eval(data, _model, args.top_k)
