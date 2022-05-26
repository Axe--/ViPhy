import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict
from utils import read_json, save_json

# Map attributes to color
ATTR2COLOR = {'gold': 'yellow', 'golden': 'yellow', 'blond': 'yellow', 'blonde': 'yellow',
              'wooden': 'brown', 'tan': 'brown', 'beige': 'yellow brown', 'bronze': 'brown',
              'grey': 'gray', 'silver': 'gray', 'metal': 'gray', 'copper': 'brown',
              'peach': 'yellow pink', 'cream': 'yellow', 'violet': 'purple',
              'maroon': 'red', 'turquoise': 'blue', 'teal': 'blue green'}

# Primary Colors
COLOR_SET = ['red', 'orange', 'yellow', 'brown', 'green',
             'blue', 'purple', 'pink', 'gray', 'black', 'white']


def find_closest_color(rgb):
    """
    Compute closest color in RGB space (L2 distance).

    >>> col = [155, 155, 155]
    >>> find_closest_color(col)
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
    Given model predictions {"colors": freq}, compute the
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


def _visualize(df: pd.DataFrame):
    df.plot.bar(x='object', stacked=True, color=COLOR_SET,
                edgecolor="lightgray", figsize=(40, 24), rot=90)
    plt.xticks(fontsize=16)
    plt.legend(loc=(1.02, 0))
    plt.show()


if __name__ == '__main__':
    # Read Raw Color data
    raw_color_data = read_json('./obj_raw_colors_90.json')

    data = []
    for obj in raw_color_data:
        # Object Name
        out = dict(object=obj['name'])

        # Primary Colors
        c2f = _map_pred_to_color(obj['raw_colors'])

        # Normalize
        total = sum(c2f.values())
        c2f = {col: freq/total for col, freq in c2f.items()}

        out = {**out, **c2f}

        # Append
        data.append(out)

    # Stacked Bar Plot
    data = pd.DataFrame(data)

    _visualize(data)
