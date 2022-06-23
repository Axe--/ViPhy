import time
import json
import torch
import pickle
from torch.distributions import Categorical
from typing import List, Dict, Union, Any
from collections import OrderedDict


def compute_entropy(freqs: List[float]) -> float:
    """The entropy is computed with natural logarithm (`e`)"""
    # tensor
    freqs = torch.tensor(freqs)

    # entropy
    entropy = Categorical(freqs).entropy().item()

    return entropy


def to_prob_dist(freqs: Union[List[int], Dict[str, int]]) -> \
                        Union[List[float], Dict[str, float]]:
    # check dict
    probs = list(freqs.values()) if type(freqs) == dict else freqs

    # tensor
    probs = torch.tensor(probs)

    # distribution
    probs = Categorical(probs).probs.tolist()

    if type(freqs) == dict:
        probs = {key: prob for key, prob in zip(freqs, probs)}

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
def read_txt(path: str) -> list:
    with open(path) as f:
        data = f.read().split()
    return data


def read_json(path: str) -> Union[List, Dict]:
    with open(path) as json_data:
        data = json.load(json_data)
    return data


def save_json(data, path: str, indent=None):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def save_pkl(data, path: str):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read_pkl(path: str) -> Union[List, Dict]:
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    return data


class Timer:
    def __init__(self, tag=''):
        self.tag = tag
        self.start = time.time()

    def end(self):
        time_taken = time.time() - self.start
        print('{} completed in {:.2f} secs'.format(self.tag, time_taken))
