import os
import inflect
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from PIL import Image
from os.path import join as osj
from typing import List, Dict, Tuple, Union
from spellchecker import SpellChecker
from easydict import EasyDict as edict
from color.constants import ALL_COLORS
from color.constants import IGNORE_IMAGES, IGNORE_OBJECTS, ADD_OBJECTS, DROP_QUALIFIERS
from models import POSTagger, OFA, CLIPImageText, UniCLImageText
from utils import read_json, save_json, sort_dict


class ColorDataDev:
    """
    Implements preprocessing & extraction functions
    for deriving object names from Visual Genome.
    """
    ofa_path = os.environ['OFA']
    root = os.environ['VisCS']
    emb = dict(obj_emb_path=osj(root, 'data/obj_emb.pkl'),
               img_emb_path=osj(root, 'data/img_emb.pkl'))

    def __init__(self,
                 data_dir=None,
                 pos_tag=False,
                 use_ofa=False,
                 use_im2txt_sim=False,
                 device='cpu'):
        # Data
        self.img_id2path = self.get_img_id_to_path(data_dir) if data_dir else None

        # Models
        self.tagger = POSTagger() if pos_tag else None
        self.model_ofa = OFA(self.ofa_path, device) if use_ofa else None
        self.img2text = UniCLImageText(device, **self.emb) if use_im2txt_sim else None

        # Utils
        self.inflect = inflect.engine()
        self.spell = SpellChecker()

    @staticmethod
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

    @staticmethod
    def _extend_bbox(box: List[int], im_size: Tuple[int, int], min_size=224) -> List[int]:
        """
        Increases `box` dims (w, h) to `min_size`,
        ensuring center (x,y) remain unchanged.
        """
        # unpack
        x, y, w, h = box
        im_w, im_h = im_size

        # increase size
        if w < min_size:
            x = x - (min_size - w) / 2
            w = min_size

        if h < min_size:
            y = y - (min_size - h) / 2
            h = min_size

        # to coordinates
        x1, x2 = x, x + w
        y1, y2 = y, y + h

        # clip to img dims
        x1, x2 = np.clip([x1, x2], 0, im_w).astype(int)
        y1, y2 = np.clip([y1, y2], 0, im_h).astype(int)

        # to box
        box = [x1, y1, x2 - x1, y2 - y1]

        return box

    @staticmethod
    def _crop_region(im: Image, box: List[int]) -> Image:
        """
        Returns cropped region from image.
        """
        x, y, w, h = box

        left, top = x, y
        right, down = x + w, y + h

        box = (left, top, right, down)

        return im.crop(box)

    def _get_best_subtype(self, im_path: str, bbox: List[int], anchor: str, candidates: List[str]) -> str:
        """
        Selects the best subtype of the anchor object (from the candidate set).

        We first use image as query to filter candidates (which score better than anchor),
        and then incorporate cropped region to query the best candidate.

        [anchor + candidates + Image] --> [selected + Region] --> best subtype
        """

        # Load image
        image = Image.open(im_path)
        im_id = im_path.split('/')[-1].replace('.jpg', '')

        # Crop bbox
        bbox = self._extend_bbox(bbox, image.size, min_size=224)
        crop = self._crop_region(image, bbox)

        # If noisy annotation: set best = anchor
        if 0 in crop.size:
            return anchor

        # Filter subtype candidates using image query
        selected = self.img2text.inference(None, anchor, candidates, img_id=im_id)
        selected = list(selected)

        # Select the best subtype using cropped region query
        best = self.img2text.inference(crop, anchor, selected, pick_best=True)
        best = list(best)[0]

        return best

    def map_region_obj_to_best_subtype(self,
                                       data_regs: List[Dict[str, Union[int, List]]],
                                       data_pos: List[Dict[str, Union[int, List]]],
                                       data_obj2subs: Dict[str, Dict[str, int]]) -> List[Dict[str, Union[int, List]]]:
        """
        Computes the best object subtype, for all objects mentioned in the region captions.

        :param data_regs: image region annotation (json)
        :param data_pos: pos-tagged region captions (json)
        :param data_obj2subs: object-to-subtype mapping
        :return: object subtypes for regions, for all images
        """
        def _get_bbox(region_annot: Dict) -> List[int]:
            return [region_annot['x'], region_annot['y'],
                    region_annot['width'], region_annot['height']]

        def _drop_color(_obj: str) -> str:
            # +1-grams
            if len(_obj.split()) > 1:
                for col in ALL_COLORS:
                    qualifiers = _obj.split()[:-1]
                    # remove
                    if col in qualifiers:
                        _obj = _obj.replace(col + ' ', '')
            return _obj

        def _ignore_object(_obj: str) -> bool:
            """Check if object should be skipped"""
            if _obj in IGNORE_OBJECTS:
                return True
            # check within subtypes as well
            for o in IGNORE_OBJECTS:
                subs = data_obj2subs.get(o, [])
                if _obj in subs:
                    return True
            return False

        def _retrieve_subtypes(obj_name: str, obj2subs: Dict[str, Dict[str, int]]):
            """
            Searches `obj_name` to find the most specific name
            by incrementally dropping qualifiers.
            """
            obj_words = obj_name.split()

            while len(obj_words) > 0:
                obj_name = ' '.join(obj_words)

                subs = obj2subs.get(obj_name, False)
                if subs:
                    return dict(object=obj_name, subtypes=list(subs))
                # drop qualifier
                obj_words.pop(0)
            return False

        region_obj2subtype = []

        for i in tqdm(range(len(data_regs))):
            im_id = data_regs[i]['image_id']
            # Skip corrupt images
            if im_id in IGNORE_IMAGES:
                continue

            im_path = self.img_id2path[im_id]
            regions = data_regs[i]['regions']
            sent_tags = data_pos[i]['pos_tags']

            image_regions = []

            for region, sent in zip(regions, sent_tags):
                # Retrieve names
                if 'of' in sent or 'has' in sent:
                    obj_names = self._concat_obj_part(sent)
                else:
                    obj_names = self._get_obj_names(sent)

                # Remove color words
                obj_names = [_drop_color(o) for o in obj_names]
                # Remove specified objects
                obj_names = [o for o in obj_names if not _ignore_object(o)]

                # Get bbox
                bbox = _get_bbox(region)

                # Compute best subtype
                obj2subtype = {}

                for obj in obj_names:
                    # get subtype names
                    res = _retrieve_subtypes(obj, data_obj2subs)

                    if res:
                        # unpack
                        obj = res['object']
                        subtypes = res['subtypes']

                        # has subtypes
                        if len(subtypes) > 1:
                            best_subtype = self._get_best_subtype(im_path, bbox, obj, subtypes)
                        else:
                            best_subtype = obj

                        obj2subtype[obj] = best_subtype

                # Add object-subtypes for the region
                image_regions.append(obj2subtype)

            image_regions = dict(image_id=im_id,
                                 ros=image_regions)

            region_obj2subtype.append(image_regions)

        return region_obj2subtype

    def _concat_obj_part(self, sent: Dict[str, str]) -> List[str]:
        """
        Extracts 'part-whole' object relations from POS-tagged sentence.

        We consider "HAS" & "OF" relations to determine object parts.

        >>> In:  {'piano': 'NN', 'keyboard': 'NN', 'has': 'IN', 'keys': 'NNS'}
        >>> Out: "piano keyboard key"
        >>> In: {'box': 'NN', 'of': 'IN', 'matches': 'NNS'}
        >>> Out: "match box"
        >>> In:  {'wheel': 'NN', 'of': 'IN', 'a': 'DT', 'bike': 'NN'}
        >>> Out: "bike wheel"

        :param sent: paired word-tag sentence
        :return: merged object-part name
        """
        # objects (nouns)
        obj_names = self._get_obj_names(sent)

        if len(obj_names) <= 1:
            return obj_names

        # relation keyword
        if 'of' in sent:
            rel = ' of '
        else:
            rel = ' has '

        # split about keyword
        sent_str = ' '.join(sent)
        try:
            seg1, seg2 = sent_str.split(rel)
        except ValueError:
            return obj_names

        part, whole = '', ''
        for o in obj_names:
            if rel == ' of ':
                if (o + rel in seg1 + rel) and part == '':
                    part = o
                if o in seg2 and whole == '':
                    whole = o
            else:
                if (o + rel in seg1 + rel) and whole == '':
                    whole = o
                if o in seg2 and part == '':
                    part = o

        # merge
        object_name = whole + ' ' + part

        return [object_name]

    def _get_obj_names(self, sent: Dict[str, str]) -> List[str]:
        """
        Extracts nouns from POS-tagged sentence.

        >>> ex = {'wall': 'NN', 'clock': 'NN', 'above': 'IN', 'desk': 'NN'}
        >>> self._get_obj_names(ex)
        >>> ['wall clock', 'desk']

        :param sent: paired word-tag sentence
        :return: object names
        """
        def _singular(_word: str) -> str:
            out = self.inflect.singular_noun(_word)
            return out if out else _word

        def _clean(s: str):
            s = s.replace('.', '')
            s = s.replace("'s", '')
            s = s.replace("`s", '')
            s = s.replace('"', '')
            s = s.replace("\\", '')
            s = s.replace('/', '')
            s = s.replace(',', '')
            return s

        _objs = []
        prev_tag = ''
        for word, tag in sent.items():
            # If word is Noun
            if 'NN' in tag:
                # clean
                word = _clean(word)

                # skip noise
                if len(word) < 2:
                    continue

                # plural
                word = _singular(word) if tag == 'NNS' else word

                # consecutive NNs
                if 'NN' in prev_tag:
                    # retrieve the last word
                    last = _objs[-1]
                    word = last + ' ' + word
                    # remove last word
                    _objs = _objs[:-1]

                # append new word
                _objs += [word]

            # Update previous tag
            prev_tag = tag

        return _objs

    @staticmethod
    def _get_subtypes(objects: Dict[str, int], obj_name: str) -> Dict[str, int]:
        """
        Computes subtypes for the given `obj_name` & retains sub-types above `min_count`.

        We only consider a subtype (e.g. baseball) if its alias (base ball) exists,
        and then collapse them into a single name (higher freq).
        """
        subtypes = {}

        for obj, freq in objects.items():                   # obj_name = 'ball'
            # Check `obj_name` is the last word
            if obj.split()[-1] == obj_name:                 # obj = 'base ball'
                # build alias
                obj_alias = ''.join(obj.split())            # obj_alias = 'baseball'
                freq_alias = objects.get(obj_alias, None)

                # If alias exists, then collapse names
                if freq_alias:
                    freq_total = freq + freq_alias

                    if freq_alias > freq:
                        subtypes[obj_alias] = freq_total
                    else:
                        subtypes[obj] = freq_total

                # Else retain the original name
                else:
                    subtypes[obj] = freq

        # Sort (descending)
        subtypes = sort_dict(subtypes, by='v', reverse=True)

        return subtypes

    def subtypes_from_objects(self, objects: Dict[str, int], obj_min: int, sub_min: int) -> Dict[str, Dict[str, int]]:
        """
        Extracts subtypes for all objects. \n
        For a given class name, we define its subtypes as
        the objects that contain the class as the last word.

        e.g. ball --> {baseball, tennis ball, snowball}

        :param objects: set of all object names (+freq)
        :param obj_min: objects below are leaves
        :param sub_min: subtypes below are excluded
        :return: object-subtypes mapping
        """
        # Remove objects below `sub_min`
        objects = {o: f for o, f in objects.items() if f >= sub_min}

        obj2subtypes = {}

        for obj, freq in tqdm(objects.items()):
            if freq >= obj_min:
                obj2subtypes[obj] = self._get_subtypes(objects, obj)

            if freq < obj_min or obj2subtypes[obj] == {}:
                obj2subtypes[obj] = {obj: freq}

        return obj2subtypes

    def objects_from_pos_tags(self, data_pos_tags: List[Dict[str, str]], min_count: int) -> Dict[str, int]:
        """
        Given POS-Tagged Region captions, builds the object set.

        :param data_pos_tags: dataset captions with POS tags
        :param min_count: objects below this frequency are excluded
        :return: object-freq dict
        """
        def _has_color(_obj: str):
            # skip 1-grams
            if len(_obj.split()) == 1:
                return False

            for col in ALL_COLORS:
                qualifiers = _obj.split()[:-1]
                if col in qualifiers:
                    return True

            return False

        def _is_noun(_obj: str):
            tag_set = ['NN', 'NNP', 'NNS', 'NNPS']
            # Tagging
            tags = self.tagger(_obj).values()
            # Check
            return all(t in tag_set for t in tags)

        def _drop_typos(object_set: Dict[str, int], _obj: str) -> Dict[str, int]:
            # check
            typos = self.spell.unknown(_obj.split())
            # drop
            if len(typos) > 0:
                object_set.pop(_obj)
            return object_set

        def _drop_objs_and_merge(obj2freq: Dict[str, int], _obj: str) -> Dict[str, int]:
            """
            Drops objects with specific qualifiers &
            merge with existing names (if exists).
            """
            _o = _obj
            qualifiers = _obj.split()[:-1]

            for q in qualifiers:
                _o = _o.replace(f'{q} ', '')
                # drop
                if q in DROP_QUALIFIERS:
                    freq = obj2freq.pop(_obj)
                    # merge
                    if _o in obj2freq:
                        obj2freq[_o] += freq
                    break

            return obj2freq

        objects = {}

        for img in tqdm(data_pos_tags):
            for sentence in img['pos_tags']:
                # Objects using POS Tags
                objs = self._get_obj_names(sentence)

                # Track frequency
                for o in objs:
                    if o not in objects:
                        objects[o] = 0
                    objects[o] += 1

        # Filter low freq
        objects = {o: f for o, f in objects.items() if f >= min_count}

        # Drop objects with more than 3 words
        objects = {o: f for o, f in objects.items() if len(o.split()) <= 3}

        # Remove words containing typos
        for obj in list(objects):
            objects = _drop_typos(objects, obj)

        # Drop objects that mention color (e.g. blue sky)
        objects = {o: f for o, f in objects.items() if not _has_color(o)}       # TODO: Review this step!

        # Retain only Noun objects
        objects = {o: f for o, f in tqdm(objects.items()) if _is_noun(o)}

        # Drop & Merge objects containing specific qualifiers
        for obj in list(objects):
            objects = _drop_objs_and_merge(objects, obj)

        # Add missing objects with default freq
        additional = {o: 200 for o in ADD_OBJECTS}

        objects = {**objects, **additional}

        # Sort by freq
        objects = sort_dict(objects, by='v', reverse=True)

        return objects

    def compute_pos_tags(self, data: List[Dict], save_path: str):
        """
        POS Tagging on Region captions, and save to disk (json).
        """
        output = []

        # Images
        for inp in tqdm(data):
            # Regions
            im_id = inp['image_id']
            pos_tags = []

            for r in inp['regions']:
                # Lowercase
                text = r['phrase'].lower()

                # POS Tag
                pos = self.tagger(text)

                pos_tags.append(pos)

            # Append
            out = dict(image_id=im_id,
                       pos_tags=pos_tags)

            output.append(out)

        # Save as JSON
        save_json(output, save_path)
        print('Done!')

    def image2color(self,
                    data_reg: List[Dict[str, Union[int, List]]],
                    data_ros: List[Dict[str, Union[int, List]]]) -> Dict[str, Dict[str, int]]:
        """
        Given region bboxes and corresponding object names (subtyped),
        computes the object color by querying the cropped region.

        :return: object to color distribution mapping
        """
        obj2colors = {}

        for i in tqdm(range(len(data_reg))):
            # Regions
            regions_bbox = data_reg[i]['regions']
            regions_obj = data_ros[i]['ros']

            # Image
            im_id = data_ros[i]['image_id']
            im_path = self.img_id2path[im_id]

            image = Image.open(im_path).convert('RGB')

            # Iterate over regions
            for b, objects in zip(regions_bbox, regions_obj):
                # Text queries & Image regions
                queries = []
                images = []
                names = []

                # Queries
                for _, obj in objects.items():
                    queries += [f"what color is the {obj}?"]
                    names += [obj]

                # Bounding box
                bbox = [b['x'], b['y'], b['width'], b['height']]
                bbox = self._extend_bbox(bbox, image.size)

                # Repeat Regions (batch inference)
                images += [self._crop_region(image, bbox)] * len(objects)

                if len(objects) > 0:
                    colors = self.model_ofa.inference(queries, images)

                    for obj, col in zip(names, colors):
                        if obj not in obj2colors:
                            obj2colors[obj] = {}
                        if col not in obj2colors[obj]:
                            obj2colors[obj][col] = 0

                        obj2colors[obj][col] += 1

        return obj2colors

    @staticmethod
    def parts_from_pos_tags(data_pos_tags: List[Dict[str, str]], obj_names: Dict[str, int], min_count: int):
        """
        Given POS-Tagged Region captions, extracts part-whole relations for objects.

        e.g. "cat has an ear" --> (cat, ear)

        :param data_pos_tags: dataset captions with POS tags
        :param obj_names: object-freq dict
        :param min_count: objects below this frequency are excluded
        :return: object-part relations
        :rtype: Dict[str, Dict[str, int]]
        """
        # I think we only need to "subtype" the Object (not needed for Part)
        pass


def _pos_tag():
    parser = argparse.ArgumentParser(description="POS Tagging (regions)")

    parser.add_argument("--inp", type=str, help="path to regions json", required=True)
    parser.add_argument("--out", type=str, help="path to output json", required=True)
    parser.add_argument("--gpu", type=int, help="cuda device ID", required=True)

    args = parser.parse_args()

    import flair
    flair.device = 'cuda:{}'.format(args.gpu)

    # Input
    data_regions = read_json(args.inp)

    # Tagger
    dev = ColorDataDev(pos_tag=True)

    # POS Tag & Save
    dev.compute_pos_tags(data_regions, save_path=args.out)


def _obj_names(min_count: int):
    data_pos = read_json('../VG/regions_pos.json')

    ddev = ColorDataDev(pos_tag=True)

    obj_names = ddev.objects_from_pos_tags(data_pos, min_count)

    save_json(obj_names, f'../data/object_names_c{min_count}.json')


def _obj_subtypes(obj_min: int, sub_min: int):
    objs = read_json('../data/object_names_c5.json')

    ddev = ColorDataDev()

    obj2subs = ddev.subtypes_from_objects(objs, obj_min, sub_min)

    save_json(obj2subs, path=f'../data/object_subtypes_o{obj_min}_s{sub_min}.json')


def _reg_obj_subtype():
    parser = argparse.ArgumentParser(description="Region Object Subtypes")

    parser.add_argument("--regions", type=str, required=True)
    parser.add_argument("--pos_tags", type=str, required=True)
    parser.add_argument("--subtypes", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--gpu", type=int, required=True)

    args = parser.parse_args()

    # args = edict()
    # args.regions = '../VG/regions/r_8.json'
    # args.pos_tags = '../VG/pos_tags/p_8.json'
    # args.subtypes = '../data/object_subtypes_o100_s10.json'
    # args.data_dir = '../VG/'
    # args.out = '../VG/ros_8.json'
    # args.gpu = 1

    # Read
    d_regs = read_json(args.regions)
    d_pos = read_json(args.pos_tags)
    d_subs = read_json(args.subtypes)

    ddev = ColorDataDev(data_dir=args.data_dir,
                        use_im2txt_sim=True,
                        device=f'cuda:{args.gpu}')

    res = ddev.map_region_obj_to_best_subtype(d_regs, d_pos, d_subs)

    save_json(res, path=args.out)


def _part_objs():
    data_pos = read_json('../VG/regions_pos.json')
    object_names = read_json('../data/object_names.json')

    ddev = ColorDataDev()

    obj_part_rels = ddev.parts_from_pos_tags(data_pos, object_names, min_count=100)

    save_json(obj_part_rels, f'../data/obj_part_relations.json')


def _im_col():
    parser = argparse.ArgumentParser(description="Region Colors")

    parser.add_argument("--regions",    type=str, required=True)
    parser.add_argument("--ros",        type=str, required=True)
    parser.add_argument("--data_dir",   type=str, required=True)
    parser.add_argument("--out",        type=str, required=True)
    parser.add_argument("--gpu",        type=int, required=True)

    args = parser.parse_args()

    # args = edict()
    # args.regions = '../VG/regions/r_8.json'
    # args.ros = '../VG/reg_obj_subtype/ros_8.json'
    # args.data_dir = '../VG/'
    # args.out = '../VG/c_8.json'
    # args.gpu = 0

    ddev = ColorDataDev(data_dir=args.data_dir,
                        use_ofa=True,
                        device=f'cuda:{args.gpu}')

    regions = read_json(args.regions)
    reg_obj_subtype = read_json(args.ros)

    res = ddev.image2color(regions, reg_obj_subtype)

    save_json(res, path=args.out)


if __name__ == '__main__':
    # _pos_tag()
    # _obj_names(min_count=5)
    # _obj_subtypes(obj_min=100, sub_min=10)
    # _reg_obj_subtype()
    _im_col()
