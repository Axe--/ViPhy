import pandas as pd
import numpy as np
from os.path import join as osj

import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Union, Any
from utils import read_json, read_csv


COLOR_LABELS = {'red': 0, 'orange': 1, 'yellow': 2, 'brown': 3, 'green': 4,
                'blue': 5, 'purple': 6, 'pink': 7, 'white': 8, 'gray': 9, 'black': 10}

SPATIAL_LABELS = {'above': 0, 'below': 1, 'same level': 2}


class ViPhyDataset(Dataset):
    """
    Dataset for Color, Spatial, and Size relations.
    """
    def __init__(self, data_dir: str, tokenizer, split: str, max_len=32, zero_shot=False):
        # Args
        self.max_len = max_len
        self.zero_shot = zero_shot
        self.is_train = (split == 'train')

        # Tokenizer
        self.tokenizer = tokenizer
        self.mask = tokenizer.mask_token

        # Model: MLM, CLM, QA
        _type = self._model_type()

        # Data -> Prompt
        if 'color' in data_dir:
            path = osj(data_dir, f'{split}.json')
            inp = read_json(path)

            self.data = self._color_prompts(inp, _type)
            self.label2idx = COLOR_LABELS

        elif 'spatial' in data_dir:
            path = osj(data_dir, f'{split}.csv')
            inp = read_csv(path)

            self.data = self._spatial_prompts(inp, _type)
            self.label2idx = SPATIAL_LABELS

        else:
            ...

    def _get_labels(self):
        label_set = []
        for d in self.data:
            label_set += d['labels']

        label_set = sorted(set(label_set))
        label_set = {label: idx for idx, label in enumerate(label_set)}

        return label_set

    def __len__(self):
        return len(self.data)

    def _model_type(self):
        tok_name = self.tokenizer.name_or_path

        if 'bert' in tok_name:
            return 'MLM'
        elif 'vilt' in tok_name:
            return 'MLM'
        elif 'gpt' in tok_name:
            return 'CLM'
        elif 'qa' in tok_name:
            return 'QA'
        else:
            raise ValueError()

    def _color_prompts(self, obj_colors: Dict[str, Dict[str, float]], model_type: str) -> List[Dict]:
        """
        Given object names & associated colors, generates textual prompts.
        """
        # Masked LM
        if model_type == 'MLM':
            if self.zero_shot:
                template = f"{'{}'} is of {self.mask} color"
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

    def _spatial_prompts(self, spatial_rels: List[Dict[str, Any]], model_type: str) -> List[Dict]:
        """
        Given spatial relations, generates textual prompts.
        """
        sign2text = {'>': 'above', '<': 'below', '=': 'same level'}

        # Masked LM
        if model_type == 'MLM':
            if self.zero_shot:
                template = f"in a {'{}'}, the {'{}'} is located {self.mask} the {'{}'}"
            else:
                template = f"in a {'{}'}, the {'{}'} is located in comparison to {'{}'}"

        # Text-to-Text
        elif model_type == 'CLM':
            template = "?"

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

    # def _size_prompts(self, spatial_rels: List[Dict[str, Any]], model_type: str) -> List[Dict]:
    #     """
    #     Given size relations, generates textual prompts.
    #     """
    #     relation_set = {'>': 'above', '<': 'below', '=': 'same size????????'}

    @staticmethod
    def _increment_char(c: str):
        return chr(ord(c) + 1)

    def _tokenize(self, text: str) -> Dict:
        text = self.tokenizer(text=text,
                              padding='max_length',
                              max_length=self.max_len,
                              truncation=True,
                              add_special_tokens=True,
                              return_attention_mask=True)

        text = {k: torch.tensor(v) for k, v in text.data.items()}

        return text

    @staticmethod
    def pack(labels: List[int]) -> str:
        """
        To pass variable no. of labels, convert to list to string.
        e.g. [1,2,3] --> '1,2,3'
        """
        labels = [str(_l) for _l in labels]
        labels = ','.join(labels)

        return labels

    def __getitem__(self, idx):
        record = self.data[idx]

        prompt = record['prompt']
        labels = record['labels']

        # Tokenize
        prompt = self._tokenize(prompt)

        # Label index
        labels = [self.label2idx[c] for c in labels]

        # Train: select a label
        if self.is_train:
            labels = np.random.choice(labels, size=1)[0]
            # to tensor
            labels = torch.tensor(labels)

        # Eval: all labels
        else:
            labels = self.pack(labels)

        model_inputs = dict(inputs=prompt, labels=labels)

        return model_inputs


if __name__ == '__main__':
    tok = AutoTokenizer.from_pretrained('bert-base-uncased')

    ds = ViPhyDataset(data_dir='./dataset/spatial', tokenizer=tok, split='val')
    dl = DataLoader(ds, batch_size=64)

    b = next(iter(dl))
    print(b)

    # for b in dl:
    #     ...
    # print('Done!')
