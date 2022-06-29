"""
Build SpatiaCS by inferring Spatial Relations from ADE-20k dataset.
"""
import json
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from os.path import join as osj
from utils import to_prob_dist
from typing import List, Dict, Union


# Erase part names from final dataset
EXCLUDE_PARTS = ['-light source', '-highlight']


def read_json(p) -> dict:
    with open(p, encoding='windows-1252') as fp:
        return json.load(fp)


class SpatialDataDev:
    """
    Pipeline for extracting Spatial Relations
    given objects in images, from ADE-20K.
    """
    def __init__(self, _dir: str):
        """
        :param _dir: path to "ADE" dir
        """
        self.data_dir = _dir

    @staticmethod
    def _index(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        arr = torch.from_numpy(arr)
        mask = torch.from_numpy(mask)
        res = arr[mask]
        return res.numpy()

    def _cluster_objects(self, path: str, k: int) -> Dict:
        """
        Given depth & annotations associated with an image,
        assigns all objects to `k` clusters.
        """
        # Annotations
        annot = read_json(path)['annotation']
        # Depth
        depth = Image.open(path.replace('.json', '_depth.jpeg'))
        # Scene
        scene_dir = '/'.join(path.split('/')[:-1])

        # to numpy
        depth = np.array(depth)

        # Cluster intervals
        c_size = int(255.0 / k)
        c_margin = {i: i * c_size for i in range(1, k + 1)}
        c_margin[k] = max(255, c_margin[k])

        for obj in annot['object']:
            # object mask
            mask = Image.open(osj(scene_dir, obj['instance_mask']))
            mask = np.array(mask)

            # exclude occluded pixels
            mask = (mask == 255) & (mask != 128)

            # Cluster
            cluster_id = None

            # if mask is not empty
            if mask.any():
                # object depth
                obj_depth = self._index(depth, mask)

                # assign cluster
                m_st = 0
                for i, m_end in c_margin.items():
                    # mean depth
                    dp_mean = int(obj_depth.mean())

                    # check cluster interval
                    if dp_mean in range(m_st, m_end):
                        cluster_id = i
                        break
                    # shift start
                    m_st = m_end

            # assign membership
            obj['cluster'] = cluster_id

        return annot

    @staticmethod
    def _pairwise_rel(o1: Dict, o2: Dict) -> str:
        # coordinates
        y1 = o1['polygon']['y']
        y2 = o2['polygon']['y']
        # centroid & upper
        c1, c2 = np.mean(y1), np.mean(y2)
        u1, u2 = np.min(y1), np.min(y2)
        # compare
        if u1 > c2:
            return '<'
        elif u2 > c1:
            return '>'
        return '='

    def _relations_in_image(self, objects: List[Dict]) -> List[Dict[str, str]]:
        """
        Computes spatial relations for all objects in the image. \n
        Only objects in the same cluster are compared.
        """
        def _with_part_name(obj: Dict) -> str:
            """
            Check if `obj` is a part, and prepend to name.
            (e.g. "knob" -> "drawer-knob")
            """
            name = obj['raw_name']
            part_of = obj['parts']['ispartof']
            if part_of != []:
                try:
                    part_of = [o['raw_name'] for o in objects
                               if part_of == o['id']][0]
                    name = part_of + '-' + name
                except IndexError:
                    pass
            return name

        objs_relations = []
        for o1 in objects:
            for o2 in objects:
                relation = {}
                # if both objects are clustered
                if o1['cluster'] and o2['cluster']:
                    # ensure same clusters
                    if o1['id'] != o2['id'] and o1['cluster'] == o2['cluster']:
                        # relation
                        relation['rel'] = self._pairwise_rel(o1, o2)

                        # names
                        relation['o1'] = _with_part_name(o1)
                        relation['o2'] = _with_part_name(o2)

                        # cluster
                        relation['cluster'] = o1['cluster']

                        objs_relations.append(relation)

        return objs_relations

    def compute_spatial_relations(self, theme: str, scene: str, k: int) -> List[Dict]:
        """
        Computes spatial relations for all images in the given scene.
        """
        path_train = osj(self.data_dir, 'training', theme, scene, '**', '*.json')
        path_val = osj(self.data_dir, 'validation', theme, scene, '**', '*.json')

        paths = glob(path_train, recursive=True)
        paths += glob(path_val, recursive=True)

        data = []
        for path in tqdm(paths):
            # Cluster Objects (by depth)
            annot = self._cluster_objects(path, k)

            # Spatial Relations
            relations = self._relations_in_image(objects=annot['object'])

            # Meta info
            for r in relations:
                r['theme'] = theme
                r['scene'] = scene
                r['fname'] = annot['filename']

            # Store
            data += relations

        return data


def _spatial_rel_freq_from_raw(raw_path: str, save_fp: str,
                               min_rel_freq: int, min_co_occur: int):
    """
    Computes relation frequencies from raw data.
    """
    def _drop_duplicates(_df):
        """
        Drops duplicated relations (o1, o2) & (o2, o1)
        """
        _df = _df.to_dict(orient='records')

        out = []
        drop = []
        for _row in tqdm(_df):
            o1 = _row['o1']
            o2 = _row['o2']
            r = _row['rel']
            f = _row['freq']

            if (o1, o2, r) not in drop:
                drop.append((o1, o2, r))

                out.append(dict(o1=o1, o2=o2,
                                rel=r, freq=f))
        return pd.DataFrame(out)

    def _get_typical(dist: Dict[str, float]) -> List[str]:
        # threshold (n=3)
        p_min = 0.2

        # filter colors
        dist = {c: p for c, p in dist.items() if p >= p_min}

        # num colors
        n = len(dist)

        if n == 2:
            p_min = 0.3

            # re-normalize
            dist = to_prob_dist(dist)

            # filter colors
            dist = {c: p for c, p in dist.items() if p >= p_min}

        # typical relations
        typ_rels = list(dist.keys())

        return typ_rels

    # Read raw relations
    raw = pd.read_csv(raw_path)

    scene_types = list(raw['scene'].unique())

    data = []
    for _scene in scene_types:
        # Filter by scene name
        df = raw[raw['scene'] == _scene]

        # Frequency of relations
        df = df.groupby(["o1", "o2", "rel"]).size()
        df = df.reset_index(name="freq")

        # Threshold by relation freq
        df = df[df['freq'] >= min_rel_freq]

        # Drop self relations
        df = df[df['o1'] != df['o2']]

        # Drop duplicates
        df = _drop_duplicates(df)

        # Scene info
        df['scene'] = _scene

        # Append
        data.append(df)

    # Stack
    data = pd.concat(data)

    # Move Relations: row -> col
    dataset = {}        # {('o1': _ 'o2': _): {rel: {'>', '=', '<'}, scene: __}}

    for row in data.to_dict(orient='records'):
        pair = (row['o1'], row['o2'])
        r = row['rel']
        if pair not in dataset:
            dataset[pair] = dict(rel={'>': 0, '=': 0, '<': 0}, scene='')

        dataset[pair]['rel'][r] = row['freq']
        dataset[pair]['scene'] = row['scene']

    data_cols = []
    for k, v in dataset.items():
        scn = v['scene']
        objs = {'o1': k[0], 'o2': k[1]}
        rels = {'>': v['rel']['>'],
                '=': v['rel']['='],
                '<': v['rel']['<']}

        # pair frequency
        freq = sum(rels.values())

        # Threshold by co-occurrence
        if freq >= min_co_occur:
            # Typical Relations
            rel_dist = to_prob_dist(rels)
            typical = _get_typical(rel_dist)
            typical = ','.join(typical)

            # meta info
            meta = {'scene': scn,
                    'freq': freq,
                    'typical': typical}

            # final row
            row = {**objs, **rels, **meta}

            data_cols.append(row)

    # Remove part names
    # TODO: EXCLUDE_PARTS

    # Save
    pd.DataFrame(data_cols).to_csv(save_fp, sep=',', index=False)


def _visualize(n_show=100, scene='bedroom'):
    # Read dataset
    df = pd.read_csv('../results/spatial.csv', sep=',')

    # Filter scene
    df = df[df['scene'] == scene]

    # Pair names
    df['pair'] = df.apply(lambda r: r['o1'] + '\n' + r['o2'], axis=1)

    # Drop others
    df = df.drop(columns=['o1', 'o2', 'scene',
                          'freq', 'typical'])
    # Select subset
    df = df.sample(n_show, replace=False)

    # Convert freq to distribution
    vis_data = []
    for d in df.to_dict(orient='records'):
        pair = d.pop('pair')
        # distribution
        dist = to_prob_dist(d)
        dist['pair'] = pair

        vis_data.append(dist)

    df = pd.DataFrame(vis_data)

    # Relations: >, =, <
    color_set = ['deepskyblue', 'gold', 'green']

    # Bar plot
    df.plot.bar(x='pair', stacked=True, color=color_set, title=scene.upper(),
                edgecolor="lightgray", figsize=(40, 24), rot=60)
    plt.xticks(fontsize=20)
    plt.legend(loc=(1.02, 0), prop={'size': 20})
    plt.margins(x=0.1)
    plt.show()


if __name__ == '__main__':
    # Save Dataset
    # _spatial_rel_freq_from_raw(raw_path='../data/spatial_raw.csv',
    #                            save_fp='../results/spatial.csv',
    #                            min_rel_freq=10, min_co_occur=100)
    # import sys; sys.exit()

    # Visualize
    theme = 'home_or_hotel'
    scenes = ['bedroom', 'living_room', 'kitchen', 'bathroom']
    for s in scenes:
        _visualize(n_show=30, scene=s)
    import sys; sys.exit()

    # Init
    ddev = SpatialDataDev(_dir='/home/axe/Datasets/ADE20K_2021_17_01/images/ADE')

    n_clusters = [2, 2, 2, 1]

    raw_relations = []
    # Iterate over scenes
    for scene, n_c in zip(scenes, n_clusters):
        # Compute all relations
        res = ddev.compute_spatial_relations(theme, scene, k=n_c)

        raw_relations += res

    # Save
    save = '../data/spatial_raw.csv'
    pd.DataFrame(raw_relations).to_csv(save, sep=',', index=False)
