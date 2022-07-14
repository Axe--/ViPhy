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

SIZE_LABELS = {'bigger': 0, 'smaller': 1}


class ViPhyDataset(Dataset):
    """
    Dataset for Color, Spatial, and Size relations.
    """
    def __init__(self, data_dir: str, tokenizer, split: str, zero_shot=False):
        # Args
        self.zero_shot = zero_shot
        self.is_train = (split == 'train')

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

    @staticmethod
    def _get_dim(data_dir):
        for dim in ['color', 'spatial', 'size']:
            if dim in data_dir:
                return dim

    @staticmethod
    def get_model_type(name):
        if 'bert' in name:
            return 'MLM'
        elif 'vilt' in name:
            return 'MLM'
        elif 'clip' in name:
            return 'MLM'
        elif 'gpt' in name:
            return 'CLM'
        elif 'opt' in name:
            return 'CLM'
        elif 'qa' in name:
            return 'QA'
        else:
            raise ValueError()
    
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
                template = f"the {'{}'} is {mask} than {'{}'}"
            else:
                template = f"the size of {'{}'} in comparison to {'{}'}"

        # Text-to-Text
        elif model_type == 'CLM':
            template = "the size of {} in comparison to {} is"

        # Question-Answer
        else:
            template = "what is the size of {} in comparison to {}? \n"
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

            # skip if both `>,<`
            if ',' in labels:
                continue

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

        model_inputs = dict(inputs=prompt, labels=labels)

        return model_inputs


if __name__ == '__main__':
    _tok = AutoTokenizer.from_pretrained('allenai/unifiedqa-t5-base')
    # _tok.pad_token = _tok.eos_token

    ds = ViPhyDataset('./dataset/size', _tok, split='train')
    dl = DataLoader(ds, batch_size=8)

    b = next(iter(dl))
    print(b)

    # for b in dl:
    #     ...
    # print('Done!')
