"""
Build SpatiaCS by inferring Spatial Relations from ADE-20k dataset.
"""
import pickle
import numpy as np
import pandas as pd
from os.path import join as osj
from im_utils import read_json


EXCLUDED = ['wall', 'floor']


def _compute_relations(img: dict):
    """
    Given image annotations comprising object (part) masks,
    computes the relative difference in position (centre-y).

    :return: {'obj1', 'y_rel', 'obj2', 'scene'}
    :rtype: list[dict]
    """
    # Image dims
    h, w, _ = img['imsize']

    # Compute y-center
    for obj in img['object']:
        # polygon
        y_pts = obj['polygon']['y']

        # centroid (%)
        y_ctr = np.mean(y_pts) / h * 100
        y_ctr = int(y_ctr)

        # add
        obj['c_y'] = y_ctr

    # Sort objects
    objects = sorted(img['object'], key=lambda x: x['c_y'])

    # Skip specific objects
    objects = [obj for obj in objects if obj['raw_name'] not in EXCLUDED]

    # Compute relations
    print(' > '.join([f"{o['raw_name']} ({o['c_y']})" for o in objects]))

    # Merge instances (same object)

    # Refine with Depth (handle "mirrors" -- maybe assign them to all 3 clusters)

    pass


def _post_process(relations, meta):
    # Manually track objects that rest on floor (since, similar level)
    # But the parts of such "floor objects" can be used.
    pass


def display_image_and_relations(img_path):
    pass


# TODO: Include "Person" as well. We can test with human parts e.g. knee-level, above waist, over head, etc.
# TODO: Should we compute `inverse depth-weighted` y-centroid ?
# TODO: Should we partition a single object into multiple depth clusters?


if __name__ == '__main__':
    # Data Directory
    root_dir = '/home/axe/Datasets'

    # Load Dataset
    with open(root_dir + '/ADE20K_2021_17_01/index_ade20k.pkl', 'rb') as f:
        ade20k = pickle.load(f)

    # Example (testing)
    p = '/home/axe/Datasets/ADE20K_2021_17_01/images/ADE/training/work_place/waiting_room/ADE_train_00019613.json'
    js = read_json(p)
    _compute_relations(js['annotation'])

    obj_part_relations = []

    # Extract Object-Part relations
    for fname, fpath, scene in zip(ade20k['filename'], ade20k['folder'], ade20k['scene']):
        # annot file
        path = osj(root_dir, fpath, fname.replace('jpg', 'json'))

        # try:
        #     annot = read_json(path)['annotation']
        # except UnicodeDecodeError:
        #     print(path)

        # obj-part relations
        # obj_part_relations += _compute_relations(annot)