import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from utils import read_json, save_json
from utils import compute_entropy, to_prob_dist
from color.constants import COLOR_SET, ATTR2COLOR


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
            # TODO: Maybe replace words via List
            raw = raw.replace(attr, color)

        # Compute Frequency
        for color in COLOR_SET:
            if color in raw:
                color2freq[color] += freq

    return color2freq


def _cluster_by_color(obj_color_dist, k: int = 4):
    """
    Given object names & associated color distribution,
    clusters all objects based on entropy into `k` sets.

    :param obj_color_dist: object color distributions
    :param k: num of clusters
    """
    # TODO: Object Overlap of UBTM as defined by `typical-fn()` vs `cluster_k=4`
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


def get_typical(_color_dist: Dict[str, float]) -> Dict[str, float]:
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


def _object_typical_colors(obj_color_dists: Dict[str, Dict[str, float]], save_fp: str):
    """
    Given object names & associated primary color distribution,
    identifies typical colors from the distribution.
    """
    # Typical Colors
    object2typical = {}

    for obj, colors in tqdm(obj_color_dists.items()):
        object2typical[obj] = get_typical(colors)

    # Save to disk
    save_json(object2typical, save_fp, indent=4)


def _merge_color_jsn(_dir: str, save_fp: str):
    from glob import glob
    from os.path import join as osj

    def _num_instances(js: Dict[str, Dict]):
        return sum(sum(col.values()) for ob, col in js.items())

    paths = sorted(glob(osj(_dir, '*.json')))

    object2colors = {}

    for path in paths:
        jsn = read_json(path)

        total = _num_instances(jsn)
        print(path.split('/')[-1], total)

        for obj, colors in jsn.items():
            if obj not in object2colors:
                object2colors[obj] = {}

            for color, freq in colors.items():
                if color not in object2colors[obj]:
                    object2colors[obj][color] = 0

                object2colors[obj][color] += freq

    print('Merged: ', _num_instances(object2colors))

    save_json(object2colors, save_fp, indent=4)


def _aggregate_subtype_colors(parent_obj: str,
                              object2color: Dict[str, Dict[str, float]],
                              object2subtypes: Dict[str, Dict[str, int]]) -> List[float]:
    """
    Computes the mean distribution of colors -- aggregated over all subtypes.
    TODO: Compute Iteratively over Subs!

    :return: object color distribution
    """

    def _get_prob(o: str) -> List[float]:
        return list(object2color[o].values())

    # Note: subtype also contains the parent!
    prob = []

    for subtype in object2subtypes[parent_obj]:
        prob += [_get_prob(subtype)]

    prob = list(np.mean(prob, axis=0))

    return prob


if __name__ == '__main__':
    # _merge_color_jsn(_dir='../VG/colors',
    #                  save_fp='../data/colors_raw.json')
    # import sys; sys.exit()

    # Args
    show_viz = False
    min_freq = 10
    save = None     # '../results/colors{}'

    # Raw Colors
    raw_data = read_json('../data/colors_raw.json')

    viz_data = []
    obj_color_data = {}

    for obj_name, raw_col in raw_data.items():
        # Primary Colors
        color_dist = _to_primary_colors(raw_col)
        freqs = list(color_dist.values())

        # Objects above threshold
        if sum(freqs) >= min_freq:
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
        save = save.format('.json')
        _object_typical_colors(obj_color_data, save)

        # Primary Colors Frequency
        save = save.replace('json', 'csv')
        pd.DataFrame(viz_data).to_csv(save, sep=',', index=False)

    # Stacked Bar Plot
    if show_viz:
        df = pd.DataFrame(viz_data)
        # TODO: Filter (~100) before Plotting
        df = df.drop(columns=['entropy', 'total'])

        _visualize(df, COLOR_SET)
        # TODO: Likewise, Plot Typical colors!
