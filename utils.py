import time
import json
import torch
from torch.distributions import Categorical


def compute_entropy(freqs: list):
    """The entropy is computed with natural logarithm (`e`)"""
    # tensor
    freqs = torch.tensor(freqs)
    # entropy
    entropy = Categorical(freqs).entropy()

    return entropy


# ------------------------------------------------
def read_json(path):
    with open(path) as json_data:
        data = json.load(json_data)
    return data


def save_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


class Timer:
    def __init__(self, tag=''):
        self.tag = tag
        self.start = time.time()

    def end(self):
        time_taken = time.time() - self.start
        print('{} completed in {:.2f} secs'.format(self.tag, time_taken))
