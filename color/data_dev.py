import requests
from tqdm import tqdm
from typing import List, Dict
from models import POSTagger, OFA
from multiprocessing import Pool, cpu_count


# Ignore corrupt images
IGNORE_IMAGE = ['2331541.jpg', '2416218.jpg', '2325661.jpg', '2330142.jpg', '2403972.jpg', '2400491.jpg',
                '2391568.jpg', '2326141.jpg', '2316472.jpg', '2369516.jpg', '2378996.jpg', '2415090.jpg',
                '2321571.jpg', '2350190.jpg', '2337155.jpg', '2404941.jpg', '2336131.jpg', '2417802.jpg',
                '2392406.jpg', '2363899.jpg', '2335991.jpg', '2340698.jpg', '2320098.jpg', '2378352.jpg',
                '2347067.jpg', '2395307.jpg', '2336852.jpg', '2372979.jpg', '2402595.jpg', '2336771.jpg',
                '2357002.jpg', '2350687.jpg', '2345238.jpg', '2405946.jpg', '2382669.jpg', '2394180.jpg',
                '2392550.jpg', '2392657.jpg', '2401212.jpg', '2328613.jpg', '2395669.jpg', '2328471.jpg',
                '2380334.jpg', '2402944.jpg', '2373542.jpg', '2394503.jpg', '2354486.jpg', '2393276.jpg',
                '2333919.jpg', '2410274.jpg', '2393186.jpg', '2415989.jpg', '2408825.jpg', '2386339.jpg',
                '2407337.jpg', '2414917.jpg', '2354293.jpg', '2408461.jpg', '2387773.jpg', '2335434.jpg',
                '2343497.jpg', '2388568.jpg', '2379781.jpg', '2387074.jpg', '2370307.jpg', '2339625.jpg',
                '2388085.jpg', '2414491.jpg', '2417886.jpg', '2401793.jpg', '2354437.jpg', '2388917.jpg',
                '2383685.jpg', '2327203.jpg', '2346773.jpg', '2390387.jpg', '2351122.jpg', '2397804.jpg',
                '2385744.jpg', '2369047.jpg', '2412350.jpg', '2346745.jpg', '2323189.jpg', '2316003.jpg',
                '2339038.jpg', '2344194.jpg', '2389915.jpg', '2406223.jpg', '2329481.jpg', '2370916.jpg',
                '2350125.jpg', '2386269.jpg', '2368946.jpg', '2414809.jpg', '2323362.jpg', '2396395.jpg',
                '2410589.jpg', '2405591.jpg', '2357389.jpg', '2370544.jpg', '2407333.jpg', '2396732.jpg',
                '2339452.jpg', '2360452.jpg', '2337005.jpg', '2329477.jpg', '2400248.jpg', '2408234.jpg',
                '2335721.jpg', '2348472.jpg', '2319788.jpg', '2319311.jpg', '2380814.jpg', '2374336.jpg',
                '2356194.jpg', '2379468.jpg', '2318215.jpg', '2335839.jpg', '2343246.jpg', '2384303.jpg',
                '2338403.jpg', '2345400.jpg', '2360334.jpg', '2397721.jpg', '2376553.jpg', '2346562.jpg',
                '2325435.jpg', '2390322.jpg', '2370333.jpg', '2320630.jpg', '2416204.jpg', '2370885.jpg',
                '2321051.jpg', '2405004.jpg', '2400944.jpg', '2379210.jpg', '2390058.jpg', '2374775.jpg',
                '2405254.jpg', '2388989.jpg', '2356717.jpg', '2416776.jpg', '2413483.jpg', '2344903.jpg',
                '2369711.jpg', '2357151.jpg', '2344765.jpg', '2338884.jpg', '2357595.jpg', '2359672.jpg',
                '2385366.jpg', '2361256.jpg', '2379927.jpg', '2407778.jpg', '2344844.jpg', '2340300.jpg',
                '2315674.jpg', '2394940.jpg', '2325625.jpg', '2355831.jpg', '2324980.jpg', '2388284.jpg',
                '2399366.jpg', '2407937.jpg', '2354396.jpg', '2397424.jpg', '2386421.jpg']

# Map attributes to color
ATTR2COLOR = {'gold': '', 'golden': '',
              'blond': '', 'blonde': '',
              'wooden': '', 'tan': '',
              'beige': '', 'bronze': '',
              'silver': '', 'metal': '',}

COLOR_SET = ['red', 'orange', 'yellow', 'brown', 'green',
             'blue', 'purple', 'pink', 'white', 'gray', 'black']


class ColorDataDev:
    """
    Implements preprocessing & extraction functions
    for deriving colors from Visual Genome.
    """
    path = '../../OFA/OFA-large'

    def __init__(self, data: List[Dict], pos_tag=False, device='cpu'):
        """
        :param data: raw regions or attributes (json)
        """
        # Args
        self.data = data

        # Models
        self.pos_tagger = POSTagger().to(device) if pos_tag else None
        self.model = OFA(self.path).to(device)

    def _search_synset(self, syn: str) -> List[dict]:
        """
        Finds image regions that contain the synset.

        :param syn: synset name
        :return: image regions
        """
        image_regions = []

        for img in self.data:
            bboxes = []
            for reg in img['attributes']:
                if syn in reg['synsets']:
                    bboxes.append(reg)

            # Image + BBoxes
            image = dict(id=img['image_id'], bboxes=bboxes)

            if len(bboxes) > 0:
                image_regions.append(image)

        return image_regions

    def map_synset_to_images(self, synsets: List[str]) -> Dict[str, List[dict]]:
        """
        Tracks images & associated regions for each object.

        :param synsets: list of object synsets
        :return: object to image regions mapping
        """
        synset2images = {}

        for syn in tqdm(synsets):
            synset2images[syn] = self._search_synset(syn)

        return synset2images

    def color_from_caption(self, cap: str):
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

    objs = list(objs.values()) * 10
    print(len(objs))

    from utils import Timer

    timer = Timer('ConceptNet:')
    outs = check_objs_in_conceptnet(objs, num_proc=32)
    timer.end()

    # print([(ob, ex) for ob, ex in zip(objs, outs)])
