import inflect
from tqdm import tqdm
from typing import List, Dict, Union
from spellchecker import SpellChecker
from color.col_utils import ALL_COLORS
from models import POSTagger, OFA, Text2TextSimilarity, Image2TextSimilarity
from utils import read_json, save_json, sort_dict


class ColorDataDev:
    """
    Implements preprocessing & extraction functions
    for deriving object names from Visual Genome.
    """
    ofa_path = '../../OFA/OFA-large'

    def __init__(self,
                 pos_tag=False,
                 use_ofa=False,
                 use_t2t_sim=False,
                 use_im2txt_sim=False,
                 device='cpu'):

        # Models
        self.tagger = POSTagger() if pos_tag else None
        self.model_ofa = OFA(self.ofa_path).to(device) if use_ofa else None
        self.text2text_sim = Text2TextSimilarity(device) if use_t2t_sim else None
        self.img2text_sim = Image2TextSimilarity(device) if use_im2txt_sim else None

        # Utils
        self.inflect = inflect.engine()
        self.spell = SpellChecker()

    @staticmethod
    def map_synset_to_images(synsets: List[str], data: List[Dict]) -> Dict[str, List[dict]]:
        """
        Tracks images & associated regions for each object.

        :param synsets: list of object synsets
        :param data: attributes file (json)
        :return: object to image regions mapping
        """
        def _search_synset(synset: str) -> List[dict]:
            """Finds image regions that contain the synset."""
            image_regions = []

            for img in data:
                bboxes = []
                for reg in img['attributes']:
                    if synset in reg['synsets']:
                        bboxes.append(reg)

                # Region: Image + BBoxes
                image = dict(id=img['image_id'], bboxes=bboxes)

                if len(bboxes) > 0:
                    image_regions.append(image)

            return image_regions

        # Iterate over all synset names
        synset2images = {}

        for syn in tqdm(synsets):
            synset2images[syn] = _search_synset(syn)

        return synset2images

    def map_object_to_image_regions(self, data_pos: List[Dict[str, str]], objects: List[str]) -> Dict[str, List[str]]:
        """
        # TODO: -----------------------------------------------

        :param data_pos:
        :param objects:
        :return:
        """
        pass

    def _get_best_subtype(self, img_path: str, caption: str, anchor: str, candidates: List[str]) -> str:
        """
        Selects the best subtype of the anchor object (from the candidate set).

        We first use image as query to filter candidates (which score better than anchor),
        and then include dense captions (objects) as query to choose the best candidate.

        TODO: Image (anchor) --> Caption (anchor) --> Region (best)
        """
        # Filter subtype candidates using image query
        selected = self.img2text_sim.inference(img_path, anchor, candidates, top_k=10)

        # Select the best subtype using caption query
        best = self.text2text_sim.inference(caption, selected)

        return best

    def map_region_obj_to_best_subtype(self,
                                       data_pos: List[Dict[str, str]],
                                       obj2subtypes: Dict[str, Dict[str, int]]) -> List[Dict[str, str]]:
        """
        ...

        :param data_pos:
        :param obj2subtypes:
        :return:
        """
        # TODO: After `get_objs()`, remove any colors mentioned in the name.split()
        def _remove_color(_obj: str):
            # ignore uni-grams
            if len(_obj.split()) > 1:
                for col in ALL_COLORS:
                    qualifiers = _obj.split()[:-1]
                    # if present
                    if col in qualifiers:
                        # remove
                        _obj = _obj.replace(col + ' ', '')
            return _obj

        pass

    def _fix_typos(self, words: List[str]) -> List[str]:
        # TODO: Handle bi-gram & tri-gram words!
        misspelled = self.spell.unknown(words)

        for word in misspelled:
            fix = self.spell.correction(word)

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
    def parts_from_pos_tags(data_pos_tags: List[Dict[str, str]], obj_names: Dict[str, int], min_count=0):
        """
        Given POS-Tagged Region captions, extracts part-whole relations for objects.

        e.g. "cat has an ear" --> (cat, ear)

        :param data_pos_tags: dataset captions with POS tags
        :param obj_names: object-freq dict
        :param min_count: objects below this frequency are excluded
        :return: object-part relations
        :rtype: Dict[str, Dict[str, int]]
        """
        def _get_part_obj_relations(sent: Dict[str, str]) -> Dict:
            """
            Extracts 'part-whole' object relations from POS-tagged sentence.

            We consider "HAS" & "OF" relations to determine object parts.

            >>> In:  {'laptop': 'NN', 'keyboard': 'NN', 'has': 'IN', 'keys': 'NNS'}
            >>> Out: {'obj': 'laptop keyboard', 'part': 'key'}
            >>> In:  {'nose': 'NN', 'of': 'IN', 'a': 'DT', 'man': 'NN'}
            >>> Out: {'obj': 'man', 'part': 'nose'}

            :param sent: paired word-tag sentence
            :return: object names
            """
            ...

        # TODO: I think we only need to "subtype" the Object (not needed for Part)
        pass

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
            # POS Tagging
            _obj = self.tagger(_obj)
            # check if all are Nouns
            tags = list(_obj.values())

            return all(t in ['NN', 'NNS'] for t in tags)

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

        # Remove typos (1-gram)
        typos = self.spell.unknown(objects)     # TODO: Fix typos (n-grams)
        [objects.pop(typo) for typo in typos if len(typo.split()) == 1]

        # Retain only Noun objects
        objects = {o: f for o, f in tqdm(objects.items()) if _is_noun(o)}

        # Drop objects that mention color (e.g. blue sky)
        objects = {o: f for o, f in objects.items() if not _has_color(o)}

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


def _region_obj_subtype(regions_fp: str, subtype_fp: str):
    ...


def _obj_subtypes(obj_min: int, sub_min: int):
    objs = read_json('../data/object_names_c5.json')

    ddev = ColorDataDev()

    obj2subs = ddev.subtypes_from_objects(objs, obj_min, sub_min)

    save_json(obj2subs, path=f'../data/object_subtypes_o{obj_min}_s{sub_min}.json')


def _part_objs():
    data_pos = read_json('../VG/regions_pos.json')
    object_names = read_json('../data/object_names.json')

    ddev = ColorDataDev()

    obj_part_rels = ddev.parts_from_pos_tags(data_pos, object_names, min_count=100)

    save_json(obj_part_rels, f'../data/obj_part_relations.json')


def _obj_names(min_count: int):
    data_pos = read_json('../VG/regions_pos.json')

    ddev = ColorDataDev(pos_tag=True)

    obj_names = ddev.objects_from_pos_tags(data_pos, min_count)

    save_json(obj_names, f'../data/object_names_c{min_count}.json')


def _pos_tag():
    import argparse

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


if __name__ == '__main__':
    # _obj_names(min_count=5)
    _obj_subtypes(obj_min=100, sub_min=5)
    ...
