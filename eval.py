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
from typing import List, Dict, Union, Any
from transformers import AutoTokenizer
from collections import Counter
from sklearn.metrics import f1_score
from utils import read_json
from color.constants import COLOR_SET
from models import LanguageModel, UnifiedQA


def _eval_colors(dataset, model):
    is_correct = []
    preds = []

    for d in tqdm(dataset):
        # sample
        inp = d['prompt']
        true_set = d['labels']
        # inference
        pred = model(inp)
        # clean
        pred = pred.lower().strip()

        is_correct.append(pred in true_set)
        preds.append(pred)

    # Metrics
    acc = np.mean(is_correct)

    print(f'\n---- {model.name} ----\n')

    print('Accuracy** : {:.4f}'.format(acc*100))
    # print('F1-score: {:.4f}'.format(f1))
    # print('Acc @K={}: {:.4f}'.format(k, acc_top_k*100))

    print('\n', Counter(preds), '\n')


def _color_prompts(obj_colors: Dict[str, Dict[str, float]],
                   task_type: str, model_name: str) -> List[Dict]:
    """
    Given object name & associated typical colors,
    generates textual prompts for model evaluation.

    :return: prompts -- [num_objs]
    """
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


def _eval_spatial(dataset, model):
    is_correct = []
    preds = []

    for d in tqdm(dataset):
        # sample
        inp = d['prompt']
        true_set = d['labels']
        # inference
        pred = model(inp)
        # clean
        pred = pred.lower().strip()

        is_correct.append(pred in true_set)
        preds.append(pred)

    # Metrics
    acc = np.mean(is_correct)

    print(f'\n---- {model.name} ----\n')

    print('Accuracy** : {:.4f}'.format(acc * 100))
    # print('F1-score: {:.4f}'.format(f1))
    # print('Acc @K={}: {:.4f}'.format(k, acc_top_k*100))

    print('\n', Counter(preds), '\n')


def _spatial_prompts(spatial_rels: List[Dict[str, Any]],
                     task_type: str, model_name: str) -> List[Dict]:
    """
    Given object name & associated typical colors,
    generates textual prompts for model evaluation.

    :return: prompts -- [num_objs]
    """
    # Masked LM
    if task_type == 'fill-mask':
        tok = AutoTokenizer.from_pretrained(model_name)

        template = f"in a {'{}'}, the {'{}'} is located {tok.mask_token} the {'{}'}"

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
    # parser = argparse.ArgumentParser(description="Region Object Subtypes")
    #
    # parser.add_argument("--task",   type=str, required=True)
    # parser.add_argument("--model",  type=str, required=True)
    # parser.add_argument("--gpu",    type=int, required=True)
    # parser.add_argument("--eval",   type=str, required=True)
    #
    # args = parser.parse_args()

    # Args
    args = edict()
    args.task = 'fill-mask'             # 'QA'
    args.model = 'microsoft/deberta-base'    # 'allenai/unifiedqa-t5-large'
    args.eval = 'spatial'
    args.gpu = 1

    # Device
    device = f'cuda:{args.gpu}' if args.gpu >= 0 else 'cpu'

    # Model
    if 'qa-t5' in args.model:
        model = UnifiedQA(name=args.model, device=device)

    elif 'gpt-neo' in args.model:
        ...

    elif 'clip' in args.model:
        ...

    elif 'vilt' in args.model:
        ...

    elif 'visualbert' in args.model:
        ...

    else:
        model = LanguageModel(task=args.task, name=args.model, device=device)

    # Read --> Dataset --> Evaluate
    if args.eval == 'color':
        data = read_json('./results/colors.json')

        data = _color_prompts(obj_colors=data,
                              task_type=args.task,
                              model_name=args.model)

        _eval_colors(data, model)

    elif args.eval == 'spatial':
        data = pd.read_csv('./results/spatial.csv')
        data = data.to_dict(orient='records')

        data = _spatial_prompts(spatial_rels=data,
                                task_type=args.task,
                                model_name=args.model)

        _eval_spatial(data, model)

    else:
        raise NotImplemented()
