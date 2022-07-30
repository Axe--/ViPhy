"""
Evaluate the performance of prior datasets (Color & Size)
"""
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Any
from utils import read_json, read_csv, to_prob_dist


def eval_coda():
    """
    Reports the alignment (*accuracy) of
    object colors in CoDa, on ViPhy dataset.
    """
    # CoDa: Color Set
    color_set = ['black', 'blue', 'brown', 'gray', 'green',
                 'orange', 'pink', 'purple', 'red', 'white', 'yellow']

    def _read(df):
        cols = ['ngram', 'label']
        df = pd.DataFrame(df)[cols]
        df = df.drop_duplicates(subset='ngram')
        df = df.to_dict(orient='records')
        return df

    def _get_typical(_color_dist: Dict[str, float]) -> Dict[str, float]:
        p_min = 0.1
        done = False
        n = len(_color_dist)

        while not done:
            # filter colors
            _color_dist = {c: p for c, p in _color_dist.items() if p >= p_min}

            # re-normalize
            _color_dist = to_prob_dist(_color_dist)

            # if unchanged --> done
            if len(_color_dist) == n:
                done = True

            # num colors
            n = len(_color_dist)

            # threshold
            if n >= 4:
                p_min = 0.1
            elif n == 3:
                p_min = 0.2
            elif n == 2:
                p_min = 0.3
            else:
                done = True

        return _color_dist

    # Load
    ds = load_dataset('corypaik/coda')

    # Extract <color-label>
    test = _read(ds['test'])
    train = _read(ds['train'])
    val = _read(ds['validation'])

    # Merge
    dataset = train + val + test

    # CoDa Dataset
    coda_typical = {}
    coda_prob = {}

    for d in dataset:
        obj = d['ngram']
        prob = d['label']
        # distribution
        col_dist = {col: prob for col, prob in zip(color_set, prob)}

        # typical colors
        colors = _get_typical(col_dist)

        coda_typical[obj] = colors
        coda_prob[obj] = col_dist

    # ViPhy Color dataset
    viphy = read_json('./results/colors.json')

    # Relaxed Accuracy
    is_correct = []
    for obj, pred in coda_typical.items():
        if obj in viphy:
            true = viphy[obj]
            is_correct += [1 if set(pred) & set(true) else 0]

    n_objs = len(is_correct)
    acc = sum(is_correct) / n_objs

    # Confidence
    conf = []
    for obj, prob in coda_prob.items():
        if obj in viphy:
            true = viphy[obj]

            score = sum(p for c, p in prob.items() if c in true)

            conf += [score]

    conf = sum(conf) / n_objs

    print('r-Accuracy:', acc)
    print('Confidence:', conf)
    print('#Objects:', n_objs)


def eval_olmpics():
    """
    Reports the alignment (*accuracy) of
    relative object size on ViPhy dataset.
    """
    def _read(split, path):
        df = pd.read_json(path.format(split), lines=True)
        df = df.to_dict(orient='records')
        return df

    def _extract_rel(sample):
        # Extract Relation
        choices = sample['question']['choices']
        ans = sample['answerKey']
        rel = [_['text'] for _ in choices if _['label'] == ans][0]
        rel = '>' if rel == 'larger' else '<'

        # Extract Object pair
        txt = sample['question']['stem']
        txt = txt.replace('The size of a', '').replace('.', '')
        txt = txt.replace('is usually much [MASK] than the size of a', '|')
        txt = txt.split('|')
        o1, o2 = txt
        o1 = o1.strip()
        o2 = o2.strip()
        return dict(o1=o1, o2=o2, rel=rel)

    p = '../../Datasets/oLMpics/size_{}.jsonl'
    dataset = _read('train', p) + _read('dev', p)

    # Extract <o1, r, o2>
    size_model = {}
    for d in dataset:
        d = _extract_rel(d)

        size_model[(d['o1'], d['o2'])] = d['rel']

    # ViPhy Size Dataset
    viphy = read_csv('./results/size.csv')

    # Drop w/ both relations
    viphy = [d for d in viphy if ',' not in d['typical']]

    # Restructure dict
    viphy = {(d['o1'], d['o2']): d['typical'] for d in viphy}

    # Relaxed Accuracy
    is_correct = []
    for objs, pred in size_model.items():
        if objs in viphy:
            true = viphy[objs]
            is_correct += [pred == true]

    n_objs = len(is_correct)
    acc = sum(is_correct) / n_objs

    print('Accuracy**:', acc)
    print('#Obj-pairs:', n_objs)


if __name__ == '__main__':
    eval_coda()
    # eval_olmpics()
    ...
