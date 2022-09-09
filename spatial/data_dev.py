"""
Derive spatial relations from ADE-20K dataset.
"""
import json
import os

import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from os.path import join as osj
from utils import to_prob_dist, save_csv
from typing import List, Dict, Union, Any


# Exclude Part Names
ERASE_PARTS = ['light source', 'highlight', 'aperture', 'buttons', 'blade']
SKIP = ['wall', 'fs']


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

    def _img_relations(self, objects: List[Dict]) -> List[Dict[str, str]]:
        """
        Computes spatial relations for all objects in the image. \n
        Only objects in the same cluster are compared.
        """
        def _prepend_to_part(obj: Dict) -> str:
            """
            Check if `obj` is a part, and append.
            (e.g. "knob" -> "drawer knob")
            """
            name = obj['raw_name']
            parent = obj['parts']['ispartof']
            if parent != []:
                try:
                    # root parent (0-idx)
                    parent = [o['raw_name'] for o in objects
                               if parent == o['id']][0]
                    name = parent + ' ' + name
                except IndexError:
                    pass
            return name

        def _clean(o: str) -> str:
            _ = o.replace('ceiling', '')\
                .replace('upper', '')\
                .replace('lower', '').strip()
            if _ != '':
                o = ' '.join(_.split())
            return o

        objs_relations = []
        for o1 in objects:
            for o2 in objects:
                record = {}
                # part-whole & clean
                _o1 = _prepend_to_part(o1)
                _o2 = _prepend_to_part(o2)
                _o1 = _clean(_o1)
                _o2 = _clean(_o2)

                # if both objects are clustered
                if o1['cluster'] and o2['cluster']:
                    # ensure same clusters
                    if _o1 != _o2 and o1['cluster'] == o2['cluster']:
                        # objects
                        record['o1'] = _o1
                        record['o2'] = _o2

                        # relation
                        record['rel'] = self._pairwise_rel(o1, o2)

                        # cluster
                        record['cluster'] = o1['cluster']

                        objs_relations.append(record)

        return objs_relations

    def compute_relations(self, theme: str, scene: str, k: int) -> List[Dict]:
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
            relations = self._img_relations(objects=annot['object'])

            # Meta info
            for r in relations:
                r['theme'] = theme
                r['scene'] = scene
                r['fname'] = annot['filename']

            # Store
            data += relations

        return data


def _preprocess_spatial(ade_dir, save_fp):
    """
    Preprocess raw ADE-20K dataset.
    """
    # Init
    ddev = SpatialDataDev(ade_dir)

    theme = 'home_or_hotel'
    scenes = {'bedroom': 2, 'living_room': 2, 'kitchen': 2, 'home_office': 2, 'bathroom': 1}

    raw_relations = []
    # Iterate over scenes
    for scene, k in scenes.items():
        # Compute all relations
        raw_relations += ddev.compute_relations(theme, scene, k)

    raw_relations += ddev.compute_relations('work_place', 'office', k=2)

    # Save
    save_csv(raw_relations, save_fp, sep=',', index=False)


def _compute_spatial_relations(raw_path: str, save_fp: str,
                               min_rel_freq: int, min_co_occur: int):
    """
    Computes relation frequencies from raw data.
    """
    def _erase_parts(o: str):
        for p in ERASE_PARTS:
            if o.endswith(p):
                o = o.replace(p, '')
            o = o.strip()
        return o

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

    def _uni_bi_samples(_data: List[Dict]) -> List[Dict]:
        """ Drop samples with all labels `<,=,>` """
        return [d for d in _data if
                len(d['typical'].split(',')) < 3]

    def _rename_scene(s: str) -> str:
        return s.replace('home_office', 'office')

    # Read raw relations
    raw = pd.read_csv(raw_path)

    scene_types = list(raw['scene'].unique())

    # Clean up parts
    raw['o1'] = raw['o1'].apply(lambda x: _erase_parts(x))
    raw['o2'] = raw['o2'].apply(lambda x: _erase_parts(x))

    # Skip objects
    raw = raw[raw['o1'].map(lambda o: o not in SKIP)]
    raw = raw[raw['o2'].map(lambda o: o not in SKIP)]

    data = []
    for _scene in scene_types:
        # Filter by scene name
        df = raw[raw['scene'] == _scene]

        # Frequency of relations
        df = df.groupby(["o1", "o2", "rel"]).size()
        df = df.reset_index(name="freq")

        # Threshold by "per-relation" freq
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
        scn = _rename_scene(scn)

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

    # Retain uni- & bi- label samples
    data_cols = _uni_bi_samples(data_cols)

    # Save
    save_csv(data_cols, save_fp, sep=',', index=False)


if __name__ == '__main__':
    # Raw Dataset
    # _preprocess_spatial(ade_dir=os.environ['ADE'],
    #                     save_fp='../data/spatial_raw.csv')

    # Final Dataset
    _compute_spatial_relations(raw_path='../data/spatial_raw.csv',
                               save_fp='../results/spatial.csv',
                               min_rel_freq=10, min_co_occur=100)
