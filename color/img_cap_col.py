"""
Constructs dataset for Img+Txt evaluation,
with VG captions as ground-truth color.
"""
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple, Union, Any
from color.col_utils import COLOR_SET
from color.constants import IGNORE_IMAGES
from utils import read_json, save_json


def _caption_with_color(data_pos: List[Dict[str, Union[int, List]]],
                        data_ros: List[Dict[str, Union[int, List]]],
                        data_box: List[Dict[str, Union[int, List]]]) -> List[Dict[str, Any]]:
    """
    Identifies image region captions mentioning object color.
    The colors are assigned with predefined template -- `COLOR OBJ`.

    Output: {'img_id': _, 'obj': O, 'color': C, 'bbox': [x, y, w, h]}

    :param data_pos: pos-tagged region captions
    :param data_ros: object-subtype names in captions
    :param data_box: region bbox coordinates
    :return: object-color mapping per caption (+bbox)
    """
    def _replace_attr(pos: Dict[str, str]):
        attr2col = {'grey': 'gray', 'silver': 'gray',
                    'tan': 'brown', 'wooden': 'brown'}
        for attr, col in attr2col.items():
            pos = {col if w == attr else w: tag
                   for w, tag in pos.items()}
        return pos

    def _has_color(pos: Dict[str, str]):
        # check if color
        for c in COLOR_SET:
            for w in pos:
                if w == c:
                    return True
        return False

    def _get_obj_color(pos: Dict[str, str], objs: Dict[str, str]) -> Dict[str, List[str]]:
        """ Finds object-color pair """
        cap = ' '.join(pos)
        colors = {}
        for c in COLOR_SET:
            for w in pos:
                if w == c:
                    i = cap.find(c)
                    colors[i] = c
        res = {}
        for o in objs:
            o_i = cap.find(o)
            for c_i, col in list(colors.items()):
                if c_i < o_i:
                    if o not in res:
                        res[o] = []
                    # save obj-col
                    res[o] += [col]
                    # drop assigned color
                    colors.pop(c_i)
        return res

    dataset = []
    for dp, dr, db in tqdm(list(zip(data_pos, data_ros, data_box))):
        # skip corrupt images
        if dp['image_id'] in IGNORE_IMAGES:
            continue

        for caption, objects, regions in zip(dp['pos_tags'], dr['ros'], db['regions']):
            # image & bbox
            _id = dp['image_id']
            box = regions['bbox']

            # threshold by area
            if box[2]*box[3] < 224:
                continue

            # clean up
            caption = _replace_attr(caption)

            obj2col = {}
            # If one object + color(s)
            if _has_color(caption) and len(objects) == 1:
                obj2col = _get_obj_color(caption, objects)
                # replace object w/ subtype
                obj2col = {objects[o]: c for o, c in obj2col.items()}

            # Dataset
            for o, c in obj2col.items():
                out = dict(object=o, color=c, img_id=_id, bbox=box)
                dataset += [out]

    return dataset


def _run_cap_col():
    # Read Captions + Subtypes + BBox
    path = '../../../Datasets/Visual_Genome/region_{}.json'

    jsn_pos = read_json(path.format('pos'))
    jsn_ros = read_json(path.format('ros'))
    jsn_bbox = read_json('../data/obj_size_k_5.json')

    # Captions with Color
    col_caps = _caption_with_color(jsn_pos, jsn_ros, jsn_bbox)

    save_json(col_caps, path='../data/img_color.json')


def _build_dataset(inp_path, save_fp, N=10):
    """
    Constructs the Caption-Color dataset (splits).
    """
    def _to_records(_data: Dict[str, List[Dict]], _obj_split: List[str]) -> List[Dict[str, Any]]:
        dataset = []
        for name in _obj_split:
            if name in _data:
                # all instances
                inst = _data[name]

                # sample `n` per object
                inst = np.random.choice(inst, size=N, replace=False)

                # Dataset
                for i in inst:
                    sample = dict(object=name,
                                  color=i['color'],
                                  img_id=i['img_id'],
                                  bbox=i['bbox'])
                    dataset += [sample]

        return dataset

    # Raw Caption-Color data
    jsn = read_json(inp_path)

    # Group by: object -> [{col, img, box}]
    data = {}
    for d in tqdm(jsn):
        obj = d.pop('object')

        if obj not in data:
            data[obj] = []

        data[obj] += [d]

    # Filter: Freq >= N
    data = {obj: inst for obj, inst in data.items() if len(inst) >= N}

    # Split by `color` dataset objects
    train = list(read_json('../dataset/color/train.json'))
    test = list(read_json('../dataset/color/test.json'))
    val = list(read_json('../dataset/color/val.json'))

    # Prepare dataset
    train = _to_records(data, train)
    test = _to_records(data, test)
    val = _to_records(data, val)

    # Save
    save_json(train, save_fp.format('train'))
    save_json(test, save_fp.format('test'))
    save_json(val, save_fp.format('val'))


if __name__ == '__main__':
    # _run_cap_col()
    _build_dataset(inp_path='../data/img_color.json', N=10,
                   save_fp='../dataset/img_color/{}.json')
