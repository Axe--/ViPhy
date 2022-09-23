import os
import numpy as np
from PIL import Image
from glob import glob
from os.path import join as osj

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers import AutoFeatureExtractor as AutoFE
from transformers import DataCollatorForLanguageModeling as DataCollatorMLM
from typing import List, Dict, Tuple, Union, Any, Optional
from utils import read_json, read_csv, read_txt


COLOR_LABELS = {'red': 0, 'orange': 1, 'yellow': 2, 'brown': 3, 'green': 4,
                'blue': 5, 'purple': 6, 'pink': 7, 'white': 8, 'gray': 9, 'black': 10}

SPATIAL_LABELS = {'above': 0, 'below': 1, 'same level': 2}

SIZE_LABELS = {'bigger': 0, 'smaller': 1}


class ViPhyDataset(Dataset):
    """
    Dataset for Color, Spatial, and Size relations.
    """
    VG = os.environ['VG']
    DIMS = ['img_color', 'color', 'spatial', 'size']

    def __init__(self, data_dir: str, model_name: str, tokenizer, split: str, zs=False):
        # Args
        self.zero_shot = zs
        self.vis_feat = None
        self.is_train = ('train' in split)

        # Tokenizer
        self.tokenizer = tokenizer
        self.tok_name = tokenizer.name_or_path

        # Model: MLM, CLM, QA
        _type = self.get_model_type(self.tok_name)

        # Text-to-Text
        self.text2text = _type in ['CLM', 'QA']
        self.is_encoder_decoder = 't5' in self.tok_name

        # Data -> Prompt
        self.dim = self._get_dim(data_dir)
        
        if self.dim == 'color':
            path = osj(data_dir, f'{split}.json')
            inp = read_json(path)

            self.data = self._color_prompts(inp, _type)
            self.label2idx = COLOR_LABELS
            self.max_len = 64 if 'qa' in self.tok_name else 16

        elif self.dim == 'img_color':
            path = osj(data_dir, f'{split}.json')
            inp = read_json(path)

            self.data = self._img_color_prompts(inp)
            self.label2idx = COLOR_LABELS
            self.max_len = 16

            self.vis_feat = AutoFE.from_pretrained(model_name)

        elif self.dim == 'spatial':
            path = osj(data_dir, f'{split}.csv')
            inp = read_csv(path)

            self.data = self._spatial_prompts(inp, _type)
            self.label2idx = SPATIAL_LABELS
            self.max_len = 32

        else:
            path = osj(data_dir, f'{split}.csv')
            inp = read_csv(path)

            self.data = self._size_prompts(inp, _type)
            self.label2idx = SIZE_LABELS
            self.max_len = 32

    def _get_dim(self, data_dir):
        for dim in self.DIMS:
            if dim in data_dir:
                return dim

    @staticmethod
    def get_model_type(name: str) -> str:
        name = name.lower()
        if 'bert' in name:
            _type = 'MLM'
        elif 'vilt' in name:
            _type = 'MLM'
        elif 'clip' in name:
            _type = 'MLM'
        elif 'flava' in name:
            _type = 'MLM'
        elif 'gpt' in name:
            _type = 'CLM'
        elif 'opt' in name:
            _type = 'CLM'
        elif 'qa' in name:
            _type = 'QA'
        else:
            raise ValueError()

        return _type

    def _get_labels(self):
        label_set = []
        for d in self.data:
            label_set += d['labels']

        label_set = sorted(set(label_set))
        label_set = {label: idx for idx, label in enumerate(label_set)}

        return label_set

    def __len__(self):
        return len(self.data)

    def _color_prompts(self, obj_colors: Dict[str, Dict[str, float]], model_type: str) -> List[Dict]:
        """
        Given object names & associated colors, generates textual prompts.
        """
        # Masked LM
        if model_type == 'MLM':
            if self.zero_shot:
                mask = self.tokenizer.mask_token
                template = f"{'{}'} is of {mask} color"
            else:
                template = f"color of {'{}'}"

        # Text-to-Text
        elif model_type == 'CLM':
            template = "the color of {} is"

        # Question-Answer
        else:
            template = "what is the color of {}? \n"
            # Color choices
            idx = 'a'
            for color in COLOR_LABELS:
                template += f'({idx}) {color} '

                idx = self._increment_char(idx)

        data = []
        for obj, colors in obj_colors.items():
            # Model I/O
            prompt = template.format(obj)
            labels = list(colors.keys())

            sample = dict(object=obj, prompt=prompt, labels=labels)

            data.append(sample)

        return data

    def _img_color_prompts(self, img_colors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Given image+box, object & colors prepares textual prompts & image paths.
        """
        def _build_id2path(vg_dir: str) -> Dict[int, str]:
            _p1 = glob(osj(vg_dir, 'images_1', '*.jpg'))
            _p2 = glob(osj(vg_dir, 'images_2', '*.jpg'))

            img_paths = _p1 + _p2

            def _get_id(p: str):
                p = p.split('/')[-1]
                p = p.split('.')[0]
                return int(p)

            img_id2path = {}
            for path in img_paths:
                # get id from path
                img_id = _get_id(path)

                img_id2path[img_id] = path

            return img_id2path

        # Image ID -> Path
        id2path = _build_id2path(self.VG)

        # Masked LM
        template = "color of {}"

        data = []
        for d in img_colors:
            # Model I/O
            d['labels'] = d.pop('color')
            d['img_path'] = id2path[d['img_id']]
            d['prompt'] = template.format(d['object'])

            data.append(d)

        return data

    def _spatial_prompts(self, spatial_rels: List[Dict[str, Any]], model_type: str) -> List[Dict]:
        """
        Given spatial relations, generates textual prompts.
        """
        sign2text = {'>': 'above', '<': 'below', '=': 'same level'}

        # Masked LM
        if model_type == 'MLM':
            if self.zero_shot:
                mask = self.tokenizer.mask_token
                template = f"in a {'{}'}, the {'{}'} is located {mask} the {'{}'}"
            else:
                template = f"in a {'{}'}, the {'{}'} is located in comparison to {'{}'}"

        # Text-to-Text
        elif model_type == 'CLM':
            template = "in a {}, location of {} in comparison to {} is"

        # Question-Answer
        else:
            template = "in a {}, where is {} located in comparison to {}? \n"
            # relation choices
            idx = 'a'
            for _, choice in sign2text.items():
                template += f'({idx}) {choice} '

                idx = self._increment_char(idx)

        data = []
        for rel in spatial_rels:
            # Model I/O
            prompt = template.format(rel['scene'], rel['o1'], rel['o2'])

            if self.zero_shot and model_type == 'MLM' and '=' in rel['typical']:
                continue

            labels = rel['typical']
            for sign, text in sign2text.items():
                labels = labels.replace(sign, text)

            labels = labels.split(',')

            sample = dict(o1=rel['o1'], o2=rel['o2'], scene=rel['scene'],
                          prompt=prompt, labels=labels)

            data.append(sample)

        return data

    def _size_prompts(self, size_rels: List[Dict[str, Any]], model_type: str) -> List[Dict]:
        """
        Given size relations, generates textual prompts.
        """
        sign2text = {'>': 'bigger', '<': 'smaller'}

        # Masked LM
        if model_type == 'MLM':
            if self.zero_shot:
                mask = self.tokenizer.mask_token
                template = f"the {'{}'} is {mask} than {'{}'} in size"
            else:
                template = f"the size of {'{}'} in comparison to {'{}'}"

        # Text-to-Text
        elif model_type == 'CLM':
            template = "the size of {} in comparison to {} is"

        # Question-Answer
        else:
            template = "how is the size of {} in comparison to {}? \n"
            # relation choices
            idx = 'a'
            for _, choice in sign2text.items():
                template += f'({idx}) {choice} '

                idx = self._increment_char(idx)

        data = []
        for rel in size_rels:
            # Model I/O
            prompt = template.format(rel['o1'], rel['o2'])

            labels = rel['typical']
            for sign, text in sign2text.items():
                labels = labels.replace(sign, text)

            labels = labels.split()

            sample = dict(o1=rel['o1'], o2=rel['o2'],
                          prompt=prompt, labels=labels)

            data.append(sample)

        return data

    @staticmethod
    def _increment_char(c: str):
        return chr(ord(c) + 1)

    @staticmethod
    def pack(labels: List[Any]) -> str:
        """
        To pass variable no. of labels, convert to list to string.
        e.g. [1,2,3] --> '1,2,3'
        """
        labels = [str(_l) for _l in labels]
        labels = ','.join(labels)

        return labels

    def _create_labels(self, input_ids, prompt_len) -> torch.LongTensor:
        """
        Generates labels from token-ids for GPT* models.
        """
        def _token2id(tok: str) -> int:
            return self.tokenizer.convert_tokens_to_ids(tok)

        def _get_token_idx(query: str) -> int:
            # token str -> id
            query = _token2id(query)
            q_idx = input_ids.index(query)
            return q_idx

        labels = [-100] * self.max_len

        # Start
        start_idx = prompt_len

        # End `<\s>`
        end_idx = _get_token_idx(self.tokenizer.eos_token) + 1

        # Label IDs (ignore prompt & pad)
        labels[start_idx: end_idx] = input_ids[start_idx: end_idx]

        # To Tensor
        labels = torch.LongTensor(labels)

        return labels

    def _tokenize(self, text: str, max_len=None) -> Dict:
        if max_len is None:
            max_len = self.max_len

        text = self.tokenizer(text=text,
                              truncation=True,
                              padding='max_length',
                              max_length=max_len,
                              add_special_tokens=True,
                              return_attention_mask=True)

        text = {k: torch.tensor(v) for k, v in text.data.items()}

        return text

    def _prepare_prompt(self, prompt: str, label: str = '') -> str:
        """
        Prepares the prompt & label text.  e.g. "color of O is C </s>"
        """
        # Encoder-Decoder (T5)
        if self.is_encoder_decoder:
            # unified-qa
            if 'qa' in self.tok_name:
                text = prompt.strip()
            # vanilla
            else:
                prefix = self.dim
                text = f'{prefix}: {prompt}'

        # Decoder-only (GPT*)
        else:
            end = self.tokenizer.eos_token

            if self.is_train:
                text = f'{prompt} {label}{end}'
            else:
                text = f'{prompt}'

        return text

    def _prepare_image(self, path: str, bbox: List[int]) -> Dict:
        """ Image: read, crop, resize & tensor """
        # Read
        img = Image.open(path).convert('RGB')

        # Crop
        x, y, w, h = bbox
        coord = [x, y, x+w, y+h]

        img = img.crop(coord)

        # Resize (w=h)
        sz = self.vis_feat.size
        by = self.vis_feat.resample

        img = img.resize((sz, sz), by)

        # To Tensor
        img = self.vis_feat(img, return_tensors='pt')

        # Drop batch-dim
        img = {k: v[0] for k, v in img.data.items()}

        return img

    def __getitem__(self, idx):
        record = self.data[idx]

        prompt = record['prompt']
        labels = record['labels']

        if self.text2text:
            # Encoder-Decoder
            if self.is_encoder_decoder:
                if self.is_train:
                    # Select a label
                    labels = np.random.choice(labels, size=1)[0]

                    # Prepare
                    prompt = self._prepare_prompt(prompt)

                    # Tokenize
                    prompt = self._tokenize(prompt)

                    # Encode Label
                    labels = self._tokenize(labels, max_len=2)
                    labels = labels['input_ids']

                else:
                    # Prepare
                    prompt = self._prepare_prompt(prompt)

                    # Tokenize
                    prompt = self._tokenize(prompt)

                    # All Labels
                    labels = self.pack(labels)

            # Decoder-only
            else:
                # Train
                if self.is_train:
                    # Select a label
                    labels = np.random.choice(labels, size=1)[0]

                    # Prompt-only
                    p_len = len(self.tokenizer(prompt).input_ids)

                    # Prepare (prompt + label)
                    prompt = self._prepare_prompt(prompt, labels)

                    # Tokenize
                    prompt = self._tokenize(prompt)

                    # Label IDs
                    inp_ids = prompt['input_ids'].tolist()

                    labels = self._create_labels(inp_ids, p_len)

                # Eval
                else:
                    # Prepare (prompt-only)
                    prompt = self._prepare_prompt(prompt)

                    # Tokenize
                    prompt = self._tokenize(prompt)

                    # All Labels
                    labels = self.pack(labels)

        else:
            # Tokenize
            prompt = self._tokenize(prompt)

            # Label index
            labels = [self.label2idx[c] for c in labels]

            # Train: select label
            if self.is_train:
                labels = np.random.choice(labels, size=1)[0]

                labels = torch.tensor(labels)

            # Eval: all labels
            else:
                labels = self.pack(labels)

        # Image
        if self.vis_feat:
            image = self._prepare_image(record['img_path'], record['bbox'])
            # update
            prompt = {**prompt, **image}

        # Model I/O
        model_inputs = dict(inputs=prompt, labels=labels)

        return model_inputs


class CaptionMLMDataset(Dataset):
    """
    Caption pre-training dataset (MLM).
    """
    def __init__(self, data_dir: str, model_name: str, split: str):
        # Args
        self.max_len = 32
        self.min_cap = 2

        # Load
        self.data = self._load(data_dir, split)

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Masked LM
        self.mlm_fn = DataCollatorMLM(self.tokenizer)

    def get_collator(self):
        return self.mlm_fn

    def _load(self, _dir, split) -> List[str]:
        """
        Loads captions from specified datasets.
        """
        def _coco(fp: str) -> List[str]:
            inp = read_json(fp)
            inp = inp['annotations']

            inp = [i['caption'] for i in inp]
            return inp

        def _cc(fp: str) -> List[str]:
            inp = read_csv(fp, sep='\t', header=None)
            inp = [i[0] for i in inp]
            return inp

        def _vg(fp: str) -> List[str]:
            inp = read_txt(fp)
            return inp

        # read captions
        data = _coco(osj(_dir, 'COCO', f'{split}.json'))
        data += _cc(osj(_dir, 'CC', f'{split}.tsv'))
        data += _vg(osj(_dir, 'VG', f'{split}.txt'))

        # threshold by length
        data = [d for d in data if len(d.split()) >= self.min_cap]

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]

        # Tokenize
        text = self.tokenizer(text=text,
                              truncation=True,
                              padding='max_length',
                              max_length=self.max_len,
                              add_special_tokens=False,
                              return_special_tokens_mask=True)
        # To Tensor
        inputs = {k: torch.tensor(v) for k, v in text.data.items()}

        return inputs


def _cap_mlm():
    ds = CaptionMLMDataset('../../Datasets/Captions', 'bert-base-uncased', split='val')
    # lens = [len(ds.tokenizer.tokenize(c)) for c in ds.data]
    # print(np.mean(lens), np.std(lens)); print(sorted(lens, reverse=True)[:200])
    dl = DataLoader(ds, batch_size=4)
    b = next(iter(dl))


if __name__ == '__main__':
    # Args
    _name = 'facebook/flava-full'

    _tok = AutoTokenizer.from_pretrained(_name)
    # _tok.pad_token = _tok.eos_token

    ds = ViPhyDataset('./dataset/img_color', _name, _tok, split='train')
    dl = DataLoader(ds, batch_size=4)

    b = next(iter(dl))
    print(b)
