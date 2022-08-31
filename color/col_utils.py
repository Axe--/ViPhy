import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from utils import read_json, save_json, save_csv
from utils import compute_entropy, to_prob_dist
from color.constants import COLOR_SET, ATTR2COLOR, DROP_QUALIFIERS


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
    # initial
    p_min = 0.1
    n = len(_color_dist)
    init_dist = _color_dist.copy()
    done = False

    while not done:
        # filter colors
        _color_dist = {c: p for c, p in _color_dist.items() if p >= p_min}

        if _color_dist == {}:
            return init_dist

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
                              obj2col: Dict[str, Dict[str, float]],
                              obj2subtypes: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """
    Computes the mean distribution of colors -- aggregated over all subtypes.
    TODO: Compute Iteratively over Subs!

    :return: object color distribution
    """
    def _get_prob(o: str) -> List[float]:
        return list(obj2col[o].values())

    # Note: subtype also contains parent!
    prob = []
    for subtype in obj2subtypes[parent_obj]:
        if subtype in obj2col:
            prob += [_get_prob(subtype)]

    # Aggregate
    prob = list(np.mean(prob, axis=0))

    # To Dict
    prob = {c: p for c, p in zip(COLOR_SET, prob)}

    return prob


if __name__ == '__main__':
    # _merge_color_jsn(_dir='../VG/colors',
    #                  save_fp='../data/colors_raw.json')
    # import sys; sys.exit()

    # Args
    min_freq = 10
    show_viz = False
    save: str = None    # '../results/colors.json'

    # Inputs
    raw_data = read_json('../data/colors_raw.json')
    obj2subs = read_json('../data/object_subtypes_o100_s10.json')

    obj2colors = {}
    meta_data = []

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
            meta = dict(object=obj_name,
                        entropy=entropy,
                        total=sum(freqs))
            meta_data += [meta]

            # Object-to-Color
            obj2colors[obj_name] = color_dist

    # Aggregate Subtypes
    obj2colors = {obj: _aggregate_subtype_colors(obj, obj2colors, obj2subs) for obj in obj2colors}

    # Filter objects (prefix)
    obj2colors = {o: c for o, c in obj2colors.items()
                  if not any(o.startswith(qual) for qual in DROP_QUALIFIERS)}

    # Sort by Entropy
    meta_data = sorted(meta_data, key=lambda x: x['entropy'])
    # Round-off values
    meta_data = [{c: round(f, 4) if type(f) == float else f
                  for c, f in d.items()} for d in meta_data]

    if save:
        # Typical Primary Colors
        _object_typical_colors(obj2colors, save)

        # Meta info
        save = save.replace('.json', '_meta.csv')
        save_csv(meta_data, save, sep=',', index=False)

    # Stacked Bar Plot
    if show_viz:
        # TODO: Use `obj2colors` for Visualization!
        df = pd.DataFrame(meta_data)
        df = df.drop(columns=['entropy', 'total'])

        # TODO: Filter (~100 rows)
        _visualize(df.sample(n=20), COLOR_SET)
