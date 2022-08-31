import os
import sys
import time
import json

import pandas as pd
import torch
import pickle
import numpy as np
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


# ---------------------------------------------------------------------------
def setup_logger(parser, log_dir, file_name='train_log.txt'):
    """
    Generates log file and writes the executed python flags for the current run,
    along with the training log (printed to console). \n
    This is helpful in maintaining experiment logs (with arguments). \n
    While resuming training, the new output log is simply appended to the previously created train log file.

    :param parser: argument parser object
    :param log_dir: file path (to create)
    :param file_name: log file name
    :return: train log file
    """
    log_file_path = os.path.join(log_dir, file_name)

    log_file = open(log_file_path, 'a+')

    # python3 file_name.py
    log_file.write('python3 ' + sys.argv[0] + '\n')

    # Add all the arguments (key value)
    args = parser.parse_args()

    for key, value in vars(args).items():
        # write to train log file
        log_file.write('--' + key + ' ' + str(value) + '\n')

    log_file.write('\n\n')
    log_file.flush()

    return log_file


def print_log(msg, log_file):
    """
    :param str msg: Message to be printed & logged
    :param file log_file: log file
    """
    log_file.write(msg + '\n')
    log_file.flush()

    print(msg)


def csv2list(v, cast=str):
    assert type(v) == str, 'Converts: comma-separated string --> list of strings'
    return [cast(s.strip()) for s in v.split(',')]


def str2bool(v):
    v = v.lower()
    assert v in ['true', 'false', 't', 'f', '1', '0'], 'Option requires: "true" or "false"'
    return v in ['true', 't', '1']


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


def read_csv(path: str, **kwargs) -> Union[List, Dict]:
    csv = pd.read_csv(path, **kwargs)
    csv = csv.to_dict(orient='records')
    return csv


def save_csv(data, path: str, **kwargs):
    pd.DataFrame(data).to_csv(path, **kwargs)


class Timer:
    def __init__(self, tag=''):
        self.tag = tag
        self.start = time.time()

    def end(self):
        time_taken = time.time() - self.start
        print('{} completed in {:.2f} secs'.format(self.tag, time_taken))


def dataset_split(data, train_ratio=0.2, dev_ratio=0.1, test_ratio=0.7):
    def _shuffle(lst):
        np.random.seed(0)
        np.random.shuffle(lst)
        return lst

    # Shuffle & Split data
    _shuffle(data)

    # train set
    train_split = int(train_ratio * len(data))
    data_train = data[:train_split]

    # validation set
    rest = data[train_split:]
    dev_split = int(dev_ratio * len(data))
    data_val = rest[:dev_split]

    # test set
    rest = rest[dev_split:]
    test_split = int(test_ratio * len(data))
    data_test = rest[:test_split]

    return data_train, data_val, data_test


if __name__ == '__main__':
    # Type
    dim = 'color'

    # Read
    if dim == 'color':
        inp = read_json('./results/colors.json')
        inp = [(k, v) for k, v in inp.items()]
    elif dim == 'spatial':
        inp = read_csv('./results/spatial.csv')
    else:
        inp = read_csv('./results/size.csv')

    # Split
    train, val, test = dataset_split(inp)

    if dim == 'color':
        save_json(dict(train), './dataset/color/train.json')
        save_json(dict(val), './dataset/color/val.json')
        save_json(dict(test), './dataset/color/test.json')

    elif dim == 'spatial':
        save_csv(train, './dataset/spatial/train.csv')
        save_csv(val, './dataset/spatial/val.csv')
        save_csv(test, './dataset/spatial/test.csv')

    else:
        save_csv(train, './dataset/size/train.csv')
        save_csv(val, './dataset/size/val.csv')
        save_csv(test, './dataset/size/test.csv')

    print(len(train), len(val), len(test))
