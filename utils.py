import time
import json
import torch
from torch.distributions import Categorical
from typing import List, Dict, Union
from collections import OrderedDict


def compute_entropy(freqs: List[float]) -> float:
    """The entropy is computed with natural logarithm (`e`)"""
    # tensor
    freqs = torch.tensor(freqs)

    # entropy
    entropy = Categorical(freqs).entropy().item()

    return entropy


def to_prob_dist(freqs: List[int]) -> List[float]:
    # tensor
    freqs = torch.tensor(freqs)

    # distribution
    probs = Categorical(freqs).probs.tolist()

    return probs


def sort_dict(d: Dict, by: str = None, **kwargs) -> Dict:
    assert by in ['k', 'v', None], 'Sort `by` k / v  OR  provide `key` arg'
    if by in ['k', 'v']:
        d = sorted(d.items(), key=lambda x: x[by == 'v'], **kwargs)
    else:
        d = sorted(d.items(), **kwargs)

    d = OrderedDict(d)
    d = dict(d)
    return d


# ------------------------------------------------
def read_txt(path) -> list:
    with open(path) as f:
        data = f.read().split()
    return data


def read_json(path) -> Union[List, Dict]:
    with open(path) as json_data:
        data = json.load(json_data)
    return data


def save_json(data, path: str, indent=None):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


class Timer:
    def __init__(self, tag=''):
        self.tag = tag
        self.start = time.time()

    def end(self):
        time_taken = time.time() - self.start
        print('{} completed in {:.2f} secs'.format(self.tag, time_taken))
