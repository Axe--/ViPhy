import requests
from tqdm import tqdm
from typing import List, Dict
from models import POSTagger, OFA
from multiprocessing import Pool, cpu_count

# Map attributes to color
ATTR2COLOR = {'gold': '', 'golden': '',
              'blond': '', 'blonde': '',
              'wooden': '', 'tan': '',
              'beige': '', 'bronze': '',
              'silver': '', 'metal': '',}


class ColorDataDev:
    """
    Implements preprocessing & extraction
    functions for building color dataset.
    """
    path = '../../OFA/OFA-large'

    def __init__(self, with_image=False):
        # Args
        self.with_image = with_image

        # Models
        self.pos_tagger = POSTagger()
        self.ofa = OFA(self.path) if with_image else None

    def color_from_caption(self, cap: str) -> Dict:
        """
        Given region caption, extracts <object, color> tuple
        using POS tags and predefined templates.

        For captions without color mentions, returns None.

        Examples:
            'the clock is green in colour' --> {clock, green}
            'the back of a white car' --> {car, white}
            'a man wears grey shoes' --> {shoes, grey}
            'it is a white van' --> {van, white}
            'black colored car on street' --> {car, black}

        :return: object & color if found, else None
        """
        # Lowercase
        cap = cap.lower()
        tags = self.pos_tagger(cap)

        pass


def check_objs_in_conceptnet(objects: List[str], num_proc=None) -> List[bool]:
    """
    Implements object names querying with ConceptNet,
    using multiprocessing.

    :param objects: list of object names
    :param num_proc: num of CPU cores
    :return: flags indicating object exists
    """
    if num_proc is None:
        num_proc = cpu_count()

    with Pool(num_proc) as p:
        res = p.map_async(_query, objects).get()

    return res


def _query(obj: str) -> bool:
    obj = '_'.join(obj.split())     # e.g. 'stop sign' --> 'stop_sign'
    url = f'http://api.conceptnet.io/c/en/{obj}'
    response = requests.get(url).json()
    if 'error' in response:
        print(obj)
        return False
    return True


if __name__ == '__main__':
    objs = {1: u'person', 2: u'bicycle', 3: u'car', 4: u'motorcycle', 5: u'airplane', 6: u'bus', 7: u'train',
            8: u'truck', 9: u'boat',10: u'traffic light', 11: u'fire hydrant', 12: u'stop sign', 13: u'parking meter',
            14: u'bench', 15: u'bird', 16: u'cat', 17: u'dog', 18: u'horse', 19: u'sheep', 20: u'cow', 21: u'elephant',
            22: u'bear', 23: u'zebra', 24: u'giraffe', 25: u'backpack', 26: u'umbrella', 27: u'handbag', 28: u'tie',
            29: u'suitcase', 30: u'frisbee', 31: u'skis', 32: u'snowboard', 33: u'sports ball', 34: u'kite',
            35: u'baseball bat', 36: u'baseball glove', 37: u'skateboard', 38: u'surfboard', 39: u'tennis racket',
            40: u'bottle', 41: u'wine glass', 42: u'cup', 43: u'fork', 44: u'knife', 45: u'spoon'}

    objs = list(objs.values())
    objs = objs + objs + objs + objs
    print(len(objs))

    from utils import Timer

    timer = Timer('ConceptNet:')
    outs = check_objs_in_conceptnet(objs, num_proc=32)
    timer.end()

    # print([(ob, ex) for ob, ex in zip(objs, outs)])
