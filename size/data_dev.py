import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from glob import glob
from collections import Counter
from os.path import join as osj
from jenkspy import JenksNaturalBreaks
from typing import List, Dict, Union, Any
import matplotlib.pyplot as plt
from matplotlib.pyplot import Rectangle

from color.constants import IGNORE_IMAGES
from utils import read_json, read_csv, save_json, save_csv


# Ignore object (`endswith`)
IGNORE_OBJECTS = ['wall', 'floor', 'sky', 'cloud', 'area', 'road', 'pavement', 'street', 'sidewalk', 'sand',
                  'sea', 'ocean', 'lake', 'river', 'water', 'ground', 'grass',
                  'picture', 'photo', 'corner', 'line',
                  'hair']


class SizeDataDev:
    """
    Implements the object relative size extraction pipeline.
    """
    def __init__(self, data_dir, k: int):
        # Clustering model
        self.jnb = JenksNaturalBreaks(nb_class=k)
        self.K = k

        self.img_id2path = self.map_id2path(data_dir)

    @staticmethod
    def _compute_area(box: List[int], depth: Image) -> float:
        """
        Computes bounding box area, scaled by the mean depth.
        """
        # box -> coords
        x1, y1, w, h = box
        x2, y2 = x1+w, y1+h

        # image dims
        im_w, im_h = depth.size

        # normalize
        depth = np.array(depth) / 255
        w /= im_w
        h /= im_h

        # box mask
        mask = np.zeros_like(depth)
        mask[x1: x2, y1: y2] = 1

        # box depth
        dpt = np.mean(depth * mask)

        area = w * h * (1 - dpt)

        return area

    def _cluster_objs(self, im_id: int, regions: List[Dict]) -> List[Dict]:
        """
        Assigns cluster membership for objects in the image.
        Inserts the `cluster` flag for each region, along with scaled area.
        """
        def _cluster(arr: List[float]) -> List[int]:
            self.jnb.fit(arr)
            return self.jnb.labels_

        # Load
        path = self.img_id2path[im_id]
        dpt = Image.open(path.replace('images', 'depth'))

        # Scaled Area
        for r in regions:
            bbox = [r['x'], r['y'], r['width'], r['height']]

            r['area'] = self._compute_area(bbox, dpt)
            r['bbox'] = bbox

        # Cluster
        areas = [r['area'] for r in regions]
        n_objs = len(regions)

        clusters = _cluster(areas) if n_objs > self.K else [0] * n_objs

        for c_id, r in zip(clusters, regions):
            r['cluster'] = int(c_id)

        return regions

    def compute_obj_size(self, data_regions: List[Dict], data_ros: List[Dict], save: str):
        """
        Computes relative object size for regions in an image,
        and saves the clustered objects as json.
        """
        objects_clusters = []

        for im_reg, im_ros in tqdm(list(zip(data_regions, data_ros))):
            im_id = im_reg['image_id']

            if im_id in IGNORE_IMAGES:
                continue

            regions = im_reg['regions']
            obj_names = im_ros['ros']

            # Cluster
            regions = self._cluster_objs(im_id, regions)
            
            # Map object name -> cluster_id
            regions_clustered = []
            for r, o in zip(regions, obj_names):
                # bounding box
                box = r['bbox']

                # cluster
                c_id = r['cluster']

                # pick first object in caption
                obj = list(o.values())[0] if len(o) != 0 else 'NULL'

                region = dict(name=obj, bbox=box, cluster=c_id)

                regions_clustered += [region]

            # Clustered regions
            obj_cluster = dict(image=im_id, regions=regions_clustered)

            objects_clusters += [obj_cluster]

        # Save `object -> cluster` mapping
        save_json(objects_clusters, save)

    def compute_size_rel(self, data_obj_cluster: List[Dict], min_freq: int, save: str):
        """
        Computes size relations (>, <) and aggregates across dataset to csv.
        """
        def _ignore(o: str):
            if o == 'NULL':
                return True
            for _obj in IGNORE_OBJECTS:
                if o.endswith(_obj):
                    return True
            return False

        # relation frequency
        freq = {}
        for img in tqdm(data_obj_cluster):
            regions = img['regions']

            # iterate over clusters
            for k_low in range(self.K):
                for k_up in range(k_low+1, self.K):
                    objs_low = [r['name'] for r in regions if r['cluster'] == k_low]
                    objs_low = [o for o in objs_low if not _ignore(o)]

                    objs_up = [r['name'] for r in regions if r['cluster'] == k_up]
                    objs_up = [o for o in objs_up if not _ignore(o)]

                    # build `o1 < o2` relations
                    for o_l in objs_low:
                        for o_u in objs_up:
                            if o_l != o_u:
                                rel = (o_l, o_u)

                                if rel not in freq:
                                    freq[rel] = 0
                                freq[rel] += 1

        # include `o1 > o2` rels
        relations = []
        visited = set()
        for r in freq:
            if r not in visited:
                # mark as visited
                visited.add(r)

                f = freq[r]

                # flip pair
                r_ = (r[1], r[0])

                if r_ in freq:
                    visited.add(r_)
                    f_ = freq[r_]
                else:
                    f_ = 0

                rel = {'o1': r[1], 'o2': r[0], '>': f, '<': f_}

                relations += [rel]

        # drop relations below threshold
        relations = [r for r in relations if r['>'] + r['<'] >= min_freq]

        # complementary samples (flip): balanced
        size_dataset = []
        for r in relations:
            r_ = {'o1': r['o2'], 'o2': r['o1'],
                  '>': r['<'], '<': r['>']}
            size_dataset += [r, r_]

        # assign typical relation
        for r in size_dataset:
            gr = r['>']
            ls = r['<']
            tot = ls + gr

            if ls / tot >= 0.7:
                r['typical'] = '<'

            elif gr / tot >= 0.7:
                r['typical'] = '>'

            else:
                r['typical'] = '>,<'

        # Save as CSV
        save_csv(size_dataset, save, index=False)

    def _plot_obj_cluster(self, reg: dict, ros: dict, n_obj: int):
        # Visualize
        vis_col = ['white', 'gold', 'red', 'brown', 'black']

        def _annotate(ax, box: list, label: str, color='red', lw=1):
            x, y, w, h = box
            # text
            ax.text(x, y - 5, label, fontsize=12, color='black',
                    bbox={'facecolor': 'white', 'alpha': 0.4})
            # bbox
            rect = Rectangle((x, y), w, h, lw=lw, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        _id = reg['image_id']
        path = self.img_id2path[_id]

        # Load
        img = Image.open(path)

        plt.figure(figsize=(14, 14))
        plt.imshow(img)

        # Reference
        axes = plt.gca()

        # Cluster
        reg = self._cluster_objs(_id, reg['regions'])
        ros = ros['ros']

        # Plot
        for r, o in list(zip(reg, ros))[:n_obj]:
            # box
            bbox = r['bbox']

            # obj-area
            text = ' | '.join(o.values())
            text += f": {r['area'] * 100: .2f}"

            # cluster
            c_id = r['cluster']

            # draw
            _annotate(axes, bbox, text, color=vis_col[c_id], lw=2)

        plt.show()

    @staticmethod
    def map_id2path(vg_dir: str) -> Dict[int, str]:
        _p1 = glob(osj(vg_dir, 'images_1', '*.jpg'))
        _p2 = glob(osj(vg_dir, 'images_2', '*.jpg'))

        img_paths = _p1 + _p2

        def _path2id(p: str):
            p = p.split('/')[-1]
            p = p.split('.')[0]
            return int(p)

        img_id2path = {}
        for path in img_paths:
            # get id from path
            img_id = _path2id(path)

            img_id2path[img_id] = path

        return img_id2path


if __name__ == '__main__':
    _dir = '../../../Datasets/Visual_Genome/'

    # Data Dev
    ddev = SizeDataDev(_dir, k=5)

    # Input
    # jsn_reg = read_json(_dir + 'region_graphs.json')    # 'regions/r_1.json'
    # jsn_ros = read_json(_dir + 'region_ros.json')       # 'reg_obj_subtype/ros_1.json'

    # Size Rel
    obj_cluster_path = '../data/obj_size_k_5.json'
    # ddev.compute_obj_size(jsn_reg, jsn_ros, save=obj_cluster_path)

    jsn_obj = read_json(obj_cluster_path)
    ddev.compute_size_rel(jsn_obj, min_freq=100, save='../results/size.csv')
