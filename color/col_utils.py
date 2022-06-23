import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from utils import read_json, save_json
from utils import compute_entropy, to_prob_dist
from constants import COLOR_SET, ATTR2COLOR


def _nearest_color(rgb):
    """
    Compute closest color in RGB space (L2 distance).

    >>> col = [155, 155, 155]
    >>> _nearest_color(col)
    """
    # CSS_COLORS = ['green', 'deepskyblue', 'purple', 'hotpink', 'orange',
    #               'red', 'yellow', 'saddlebrown', 'gray', 'black', 'white']
    # COLOR2RGB = {'red': [255, 0, 0], ...}

    col_set = [[255, 0, 0], [150, 33, 77], [75, 99, 23], [45, 88, 250], [250, 0, 255]]

    colors = np.array(col_set)
    color = np.array(rgb)
    distances = np.sqrt(np.sum((colors - color) ** 2, axis=1))
    index_of_smallest = np.where(distances == np.amin(distances))
    smallest_distance = colors[index_of_smallest]
    return smallest_distance


def _to_primary_colors(raw_color_dist: Dict[str, int]) -> Dict[str, int]:
    """
    Given model predictions -- raw color frequencies,
    computes the primary color distribution.
    """
    # Primary colors
    color2freq = dict.fromkeys(COLOR_SET, 0)

    for raw, freq in raw_color_dist.items():
        # Map Raw to Primary Colors
        for attr, color in ATTR2COLOR.items():
            raw = raw.replace(attr, color)

        # Compute Frequency
        for color in COLOR_SET:
            if color in raw:
                color2freq[color] += freq

    return color2freq


def _cluster_objects_by_color_dist(obj_color_dist, k: int = 4):
    """
    Given object names & associated color distribution,
    clusters all objects based on entropy into `k` sets.

    :param obj_color_dist: object color distributions
    :param k: num of clusters
    """
    pass


def _visualize(_df: pd.DataFrame, color_set: List[str]):
    # replace color codes
    def _map(c):
        c = c.replace('white', 'whitesmoke')
        c = c.replace('blue', 'deepskyblue')
        # c = c.replace('green', 'limegreen')
        c = c.replace('gray', 'silver')
        c = c.replace('brown', 'saddlebrown')
        c = c.replace('pink', 'deeppink')
        c = c.replace('purple', 'blueviolet')
        return c

    color_set = [_map(color) for color in color_set]

    # plot
    _df.plot.bar(x='object', stacked=True, color=color_set,
                 edgecolor="lightgray", figsize=(40, 24), rot=90)
    plt.xticks(fontsize=16)
    plt.legend(loc=(1.02, 0))
    plt.show()


def _object_typical_colors(obj_color_dists: Dict[str, Dict[str, float]], save_fp: str):
    """
    Given object names & associated primary color distribution,
    identifies typical colors from the distribution.
    """
    def _get_typical(_color_dist: Dict[str, float]) -> Dict[str, float]:
        p_min = 0.1
        done = False
        n = len(_color_dist)

        while not done:
            # filter colors
            _color_dist = {c: p for c, p in _color_dist.items() if p > p_min}

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

    # Typical Colors
    object2typical = {}

    for obj, colors in tqdm(obj_color_dists.items()):
        object2typical[obj] = _get_typical(colors)

    # Save to disk
    save_json(object2typical, save_fp, indent=4)


if __name__ == '__main__':
    # Args
    show_viz = False
    save = '../results/col_90{}'

    # Raw Colors
    raw_color_data = read_json('../results/raw_90.json')

    viz_data = []
    obj_color_data = {}

    for obj_name, raw_colors in raw_color_data.items():
        # Primary Colors
        color_dist = _to_primary_colors(raw_colors)
        freqs = list(color_dist.values())

        # Distribution
        color_dist = to_prob_dist(color_dist)

        # Entropy
        entropy = compute_entropy(freqs)

        # Visualization
        out = dict(object=obj_name,
                   entropy=entropy,
                   total=sum(freqs))
        out = {**out, **color_dist}

        viz_data.append(out)

        # Object-to-Color
        obj_color_data[obj_name] = color_dist

    # Sort by Entropy
    viz_data = sorted(viz_data, key=lambda x: x['entropy'])
    # Round-off values
    viz_data = [{c: round(f, 4) if type(f) == float else f
                 for c, f in d.items()} for d in viz_data]

    if save:
        # Typical Primary Colors
        _object_typical_colors(obj_color_data,
                               save.format('.json'))

        # Primary Colors
        # save = save.format('.csv')
        # pd.DataFrame(viz_data).to_csv(save, sep=',', index=False)

    # Stacked Bar Plot
    if show_viz:
        df = pd.DataFrame(viz_data)
        df = df.drop(columns=['entropy', 'total'])

        _visualize(df, COLOR_SET)
        # TODO: Likewise, Plot Typical colors!
