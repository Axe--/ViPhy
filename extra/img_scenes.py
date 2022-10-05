"""
Computes scene name for images in Visual Genome.
"""
import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join as osj
from typing import Dict, List, Union, Any
from models import OFA
from color.constants import IGNORE_IMAGES
from utils import save_json


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
    # VG: ID to Path
    VG = os.environ['VG']
    # im2path = map_id2path(VG)

    device = 'cuda:0'
    model = OFA(os.environ['OFA'], device)

    query = 'where is this located?'

    paths = glob(osj(VG, 'images_1', '*.jpg'))
    paths += glob(osj(VG, 'images_2', '*.jpg'))

    img2scene = {}
    for path in tqdm(paths):
        name = path.split('/')[-1].split('.')[0]

        if int(name) not in IGNORE_IMAGES:
            img = Image.open(path).convert('RGB')
            img2scene[name] = model(query, img)

    # Save
    save_json(img2scene, path='./img2scene.json')
