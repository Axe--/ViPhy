import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from utils import read_json, compute_entropy, to_prob_dist
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


def _map_pred_to_color(pred: Dict[str, int]) -> Dict[str, int]:
    """
    Given model prediction {"colors": freq}, compute the
    frequencies of the primary color set.

    :param pred: raw color count
    :return: primary color frequency
    """
    # Initialize
    color2freq = dict.fromkeys(COLOR_SET, 0)

    # Iterate over raw colors
    for raw, freq in pred.items():
        # Map Raw Colors & Attributes to Primary Color Names
        for attr, color in ATTR2COLOR.items():
            raw = raw.replace(attr, color)

        # Compute Frequency
        for color in COLOR_SET:
            if color in raw:
                color2freq[color] += freq

    return color2freq


def _cluster_objects_by_color_dist(_df: pd.DataFrame, k=3):
    """
    Given object names & associated color distribution,
    clusters all objects based on entropy into `k` sets.

    :param _df: object name and primary colors as columns
    :param k: num of clusters
    :return:
    """
    # TODO: ****************************
    pass


def _visualize(_df: pd.DataFrame, color_set: List[str]):
    # Replace color codes
    def _rename(c):
        c = c.replace('white', 'whitesmoke')
        c = c.replace('blue', 'deepskyblue')
        # c = c.replace('green', 'limegreen')
        c = c.replace('gray', 'silver')
        c = c.replace('brown', 'saddlebrown')
        c = c.replace('pink', 'deeppink')
        c = c.replace('purple', 'blueviolet')
        return c

    color_set = [_rename(color) for color in color_set]

    # plot
    _df.plot.bar(x='object', stacked=True, color=color_set,
                 edgecolor="lightgray", figsize=(40, 24), rot=90)
    plt.xticks(fontsize=16)
    plt.legend(loc=(1.02, 0))
    plt.show()


if __name__ == '__main__':
    # Read Raw Color data
    raw_color_data = read_json('./temp/obj_raw_colors_90.json')

    data = []
    for obj in raw_color_data:
        # Primary Colors
        c2f = _map_pred_to_color(obj['raw_colors'])

        # Normalize
        total = sum(c2f.values())
        c2f = {col: freq/total for col, freq in c2f.items()}

        # Entropy
        entropy = compute_entropy(freqs=list(c2f.values()))

        # Store to dict
        out = dict(object=obj['name'],
                   entropy=entropy,
                   total=total)

        out = {**out, **c2f}

        # Append
        data.append(out)

    # Sort by Entropy
    data = sorted(data, key=lambda x: x['entropy'])

    # To DF
    df = pd.DataFrame(data)

    # TODO: Cluster colors
    # clusters = _cluster_objects_by_color_dist(df, k=3)

    # Remove columns
    df = df.drop(columns=['entropy', 'total'])

    # Stacked Bar Plot
    _visualize(df, COLOR_SET)

    # TODO: Save DF as CSV
    # pd.DataFrame(data).to_csv()
