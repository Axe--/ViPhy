"""
Retrieve Data from ConceptNet Web API.
"""
import os
import sys
import requests
from tqdm import tqdm
from typing import List, Dict

sys.path.append(os.environ['VisCS'])
from utils import read_json


# For Testing API
DUMMY = {1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train',
         8: u'truck', 9: u'boat', 10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter',
         14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant',
         22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie',
         29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite',
         35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket',
         40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon'}


def objects_in_conceptnet(obj_names: List[str], write_to: str):
    """
    Checks if object exists in ConceptNet, saves as txt.
    """
    def _query(obj: str) -> bool:
        obj = '_'.join(obj.split())  # e.g. 'stop sign' --> 'stop_sign'
        url = f'http://api.conceptnet.io/c/en/{obj}'
        response = requests.get(url).json()
        if 'error' in response:
            return False
        return True

    with open(write_to, 'a+') as f:
        for o in tqdm(obj_names):
            if _query(o):
                f.write(o + '\n')
                f.flush()


if __name__ == '__main__':
    # Read object / subtype names
    objs = read_json('../data/object_subtypes_o100_s10.json')
    objs = list(objs)
    print(len(objs))

    objects_in_conceptnet(objs, write_to='./objs_kb.txt')
