import argparse
import transformers
from PIL import Image
from glob import glob
from tqdm import tqdm
from os.path import join as osj
from typing import List, Dict
from collections import Counter
from utils import read_json, save_json
from constants import IGNORE_IMAGES
if transformers.__version__ == '4.18.0.dev0':
    from models import OFA


def get_img_id_to_path(_dir: str) -> Dict[int, str]:
    _p1 = glob(osj(_dir, 'images_1', '*.jpg'))
    _p2 = glob(osj(_dir, 'images_2', '*.jpg'))

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


def _box_to_coord(box: Dict[str, int]) -> List[int]:
    left = box['x']
    top = box['y']
    right = box['x'] + box['w']
    down = box['y'] + box['h']

    box = [left, top, right, down]

    return box


def _synset2object(syn: str) -> str:
    """
    Converts synset name to object name.

    >>> _synset2object("tennis_ball.n.01")
    >>> "tennis ball"
    """
    obj = syn.split('.')[0]
    obj = ' '.join(obj.split('_'))

    return obj


def main():
    parser = argparse.ArgumentParser(description="Image Region to Color")

    parser.add_argument("--data",   type=str, help="path to dataset dir", required=True)
    parser.add_argument("--inp",    type=str, help="path to ???? json", required=True)
    parser.add_argument("--out",    type=str, help="path to ???? json", required=True)
    parser.add_argument("--gpu",    type=int, help="cuda device ID", required=True)

    args = parser.parse_args()

    # TODO: Move the __main__ code, HERE!


if __name__ == '__main__':
    # Dataset Directory
    data_dir = '../../../Datasets/Visual_Genome'

    # Image ID to Path mapping
    id2path = get_img_id_to_path(data_dir)

    # Object to Images data
    obj2images = read_json('./temp/objects_to_images_100.json')

    # OFA Model
    model = OFA(path='../../OFA/OFA-large', device='cuda:1')

    for synset, images in tqdm(obj2images.items()):
        # Object Name
        name = _synset2object(synset)
        colors = []

        # Associated Images
        for img in images:
            # Skip `Corrupt Images`
            if f"{img['id']}.jpg" in IGNORE_IMAGES:
                continue

            # Path
            im_path = id2path[img['id']]

            # Read Image
            image = Image.open(im_path)

            # Image Regions
            regions = []

            for bbox in img['bboxes']:
                # Reformat: [L,T,R,D]
                coords = _box_to_coord(bbox)

                # Crop region
                reg = image.crop(coords)

                regions.append(reg)

            # Extract Color
            query = [f'what color is the {name}?'] * len(regions)

            out = model.inference(query, regions)

            # Map to Colors
            colors += out

        colors = dict(Counter(colors))

        print(f'{name}: {colors} \n')

    # TODO: Save output
    save_json()
