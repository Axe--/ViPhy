import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os.path import join as osj
from typing import Dict, List, Union
from torch import Tensor, LongTensor
from flair.data import Sentence
from collections import OrderedDict
from flair.models import SequenceTagger
from sentence_transformers import SentenceTransformer

from torchvision import transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN as M
from timm.data.constants import IMAGENET_DEFAULT_STD as S

from unicl.config import get_config
from unicl.model.model import UniCLModel
from utils import read_pkl, read_json, save_pkl, sort_dict

import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline, BertTokenizerFast
from transformers import BertTokenizer, BertForMaskedLM
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import DebertaTokenizer, DebertaPreTrainedModel
from transformers import DebertaV2Tokenizer, DebertaV2ForMaskedLM
from transformers import T5Tokenizer, T5ForConditionalGeneration as T5
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM
from transformers import CLIPTokenizerFast, CLIPTextModel
from transformers import VisualBertForPreTraining, ViltModel, ViltForMaskedLM

# DPT
if transformers.__version__ >= '4.19.0':
    from transformers import (DPTFeatureExtractor,
                              DPTForDepthEstimation,
                              DPTForSemanticSegmentation)
    from transformers import FlavaTextModel, FlavaModel

# OFA
if transformers.__version__ == '4.18.0.dev0':
    from transformers import OFATokenizer, OFAForConditionalGeneration, OFAModel
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize


def freeze(module):
    for name, param in module.named_parameters():
        param.requires_grad = False


def trainable(module):
    for name, param in module.named_parameters():
        param.requires_grad = True


# -------------------------------------------
# ************ EVALUATION MODELS ************
# -------------------------------------------


class MaskedLM(nn.Module):
    """
    Implements Masked Language Models (*BERT).
    """
    def __init__(self, name: str, num_classes: int, device='cpu', ckpt=None, vl=False, zs=False):
        super().__init__()

        # Args
        self.vl = vl
        self.name = name
        self.device = device
        self.zero_shot = zs

        # Init
        tok_name, _Tokenizer, _Model = self._init_transformer()

        # Tokenizer
        self.tokenizer = _Tokenizer.from_pretrained(tok_name)

        # Transformer (frozen)
        self.model = _Model.from_pretrained(name)
        freeze(self.model)

        # Linear Probe
        hidden_dim = self.model.config.hidden_size
        self.fc = nn.Linear(hidden_dim, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        # Load
        self._load_wts(ckpt)
        self.to(device)

    def _init_transformer(self):
        name = self.name

        # MASK model
        if self.zero_shot:
            if name.startswith('bert-'):
                _Tokenizer = BertTokenizer
                _Model = BertForMaskedLM
            elif name.startswith('CapBERT'):
                name = 'bert-base-uncased'
                _Tokenizer = BertTokenizer
                _Model = BertForMaskedLM
            elif 'visualbert' in name:
                name = 'bert-base-uncased'
                _Tokenizer = BertTokenizer
                _Model = VisualBertForPreTraining
            elif 'vilt' in name:
                name = 'bert-base-uncased'
                _Tokenizer = BertTokenizer
                _Model = ViltForMaskedLM
            elif 'flava' in name:
                name = 'bert-base-uncased'
                _Tokenizer = BertTokenizer
                _Model = FlavaTextModel
            elif 'roberta' in name:
                _Tokenizer = RobertaTokenizer
                _Model = RobertaForMaskedLM
            elif 'deberta-v2' in name:
                _Tokenizer = DebertaV2Tokenizer
                _Model = DebertaV2ForMaskedLM
            elif 'deberta' in name:
                _Tokenizer = DebertaTokenizer
                _Model = DebertaPreTrainedModel
            else:
                print(name)
                raise NotImplemented()

        # CLS model
        else:
            if 'vilt' in name:
                name = 'bert-base-uncased'
                _Tokenizer = BertTokenizerFast
                _Model = ViltModel

            elif 'clip' in name:
                _Tokenizer = CLIPTokenizerFast
                _Model = CLIPTextModel

            elif 'flava' in name:
                _Tokenizer = BertTokenizer
                _Model = FlavaModel if self.vl else FlavaTextModel

            else:
                if 'visualbert' in name or 'Cap' in name:
                    name = 'bert-base-uncased'
                _Tokenizer = AutoTokenizer
                _Model = AutoModel

        return name, _Tokenizer, _Model

    def _load_wts(self, path: str):
        if path:
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt['model_state_dict']

            state_dict_new = OrderedDict()

            for k, v in state_dict.items():
                k_ = k.replace('module.', '')
                state_dict_new[k_] = v

            self.load_state_dict(state_dict_new)

            print(f'Model successfully loaded from {path}\n')

    def _prepare_vilt(self, inputs: Dict) -> Dict:
        """ Inserts dummy visual values for ViLT. """
        if 'pixel_values' not in inputs:
            B = len(inputs['input_ids']) if not self.zero_shot else 1

            P, D = (1, self.fc.in_features)

            inputs['image_embeds'] = torch.zeros([B, P, D]).to(self.device)
            inputs['pixel_mask'] = torch.zeros([B, P]).to(self.device)

        return inputs

    def forward(self, inputs: Dict[str, Tensor], labels: Tensor = None) -> Tensor:
        # visual inputs
        if 'vilt' in self.name:
            inputs = self._prepare_vilt(inputs)

        # transformer
        if self.vl and 'flava' in self.name:
            x = self.model(**inputs)
            x = x.multimodal_output[0]
            cls_emb = x[:, 0, :]

        elif 'clip' in self.name:
            x = self.model(**inputs)
            cls_emb = x.pooler_output

        else:
            x = self.model(**inputs)[0]     # [B, L, D]
            cls_emb = x[:, 0, :]            # [B, D]

        # classifier
        logits = self.fc(cls_emb)           # [B, C]

        # eval
        if labels is None:
            return logits

        # train
        else:
            loss = self.loss_fn(logits, labels)
            return loss

    @torch.inference_mode()
    def predict(self, text: str, top_k: int) -> List[str]:
        """
        Performs Zero-Shot Inference over [MASK].

        :param text: input prompt
        :param top_k: #labels
        :return: `top-k` predictions
        """
        # Special Tokens
        cls = self.tokenizer.cls_token
        sep = self.tokenizer.sep_token
        mask = self.tokenizer.mask_token

        # Tokenize input
        text = f"{cls} {text} {sep}"

        tokens = self.tokenizer.tokenize(text)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        tokens_tensor = torch.tensor([token_ids]).to(self.device)

        mask_idx = tokens.index(mask)

        # # Predict all tokens
        if 'vilt' in self.name:
            inp = dict(input_ids=tokens_tensor)
            inp = self._prepare_vilt(inp)
            outputs = self.model(**inp)
        else:
            outputs = self.model(tokens_tensor)

        predictions = outputs[0]

        # Top-K
        probs = torch.nn.functional.softmax(predictions[0, mask_idx], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

        preds_top_k = []
        for i, pred_idx in enumerate(top_k_indices):
            pred = self.tokenizer.convert_ids_to_tokens([pred_idx])[0]
            pred = pred.replace('Ġ', '').replace('ġ', '')
            preds_top_k.append(pred)

        return preds_top_k


class Text2TextLM(nn.Module):
    """
    Implements Text-to-Text Language Models (GPT*, T5)
    """
    def __init__(self, name: str, device='cpu', ckpt=None, zs=False):
        super().__init__()

        # Args
        self.name = name
        self.device = device
        self.zero_shot = zs

        # Init
        tok_name, _Tokenizer, _Model = self._init_transformer()

        # Tokenizer
        self.tokenizer = _Tokenizer.from_pretrained(tok_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token
        eos_id = self.tokenizer.eos_token_id

        # Transformer
        self.model = _Model.from_pretrained(name, pad_token_id=eos_id)

        # Freeze Backbone
        freeze(self.model)

        # Train LM Head
        trainable(self.model.lm_head)

        # Load
        self._load_wts(ckpt)
        self.to(device)

    def _init_transformer(self):
        name = self.name

        if 'gpt2' in name:
            _Tokenizer = GPT2Tokenizer
            _Model = GPT2LMHeadModel
        elif 'neo' in name:
            _Tokenizer = GPT2Tokenizer
            _Model = GPTNeoForCausalLM
        elif 'opt' in name:
            _Tokenizer = ...
            _Model = ...
        else:
            _Tokenizer = ...
            _Model = ...

        return name, _Tokenizer, _Model

    def _load_wts(self, path: str):
        if path:
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt['model_state_dict']

            state_dict_new = OrderedDict()

            for k, v in state_dict.items():
                k_ = k.replace('module.', '')
                state_dict_new[k_] = v

            self.load_state_dict(state_dict_new)

            print(f'Model successfully loaded from {path}\n')

    def _generate_from_prompt(self, inputs: Dict[str, Tensor]) -> List[str]:
        """
        Generates prediction from input prompt,
        by clipping prompt from the output.

        :param inputs: prompt text [B, L]
        :return: predicted labels [B]
        """
        def clip(prompt: str, gen: str) -> str:
            p_len = len(prompt.split())
            gen = gen.split()[p_len:]
            gen = ' '.join(gen)
            if len(gen) == 0:
                gen = '*'   # prevent div by zero
            return gen

        def decode(_ids) -> List[str]:
            return self.tokenizer.batch_decode(_ids, skip_special_tokens=True)

        out_b = self.model.generate(**inputs, max_new_tokens=4)

        out_b = decode(out_b)
        inp_b = decode(inputs['input_ids'])

        out = []
        for i, o in zip(inp_b, out_b):
            out += [clip(i, o)]

        return out

    def forward(self, inputs: Dict[str, Tensor], labels: Tensor = None) -> Union[Tensor, List[str]]:
        # pack
        inputs['labels'] = labels

        # eval
        if labels is None:
            out = self._generate_from_prompt(inputs)
            return out

        # train
        else:
            out = self.model(**inputs)     # [B, L, D]
            return out.loss


class UnifiedQA(nn.Module):
    """
    Implements Unified-QA (T5) model.
    """
    def __init__(self, name: str, device='cpu', ckpt=None):
        super().__init__()
        # Args
        self.device = device
        self.name = name

        # Tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(name)

        # Transformer
        self.model = T5.from_pretrained(name, low_cpu_mem_usage=True)

        # Freeze Backbone
        freeze(self.model)

        # Train LM Head
        trainable(self.model.lm_head)

        # Load
        self._load_wts(ckpt)

        # Device
        if '11b' in name and 'cuda' in device:
            # TODO: model.half() ?
            self.model.parallelize()
            device = 'cuda:0'
        else:
            self.model.to(device)

    def _load_wts(self, path: str):
        if path:
            ckpt = torch.load(path, map_location='cpu')
            state_dict = ckpt['model_state_dict']

            state_dict_new = OrderedDict()

            for k, v in state_dict.items():
                k_ = k.replace('module.', '')
                state_dict_new[k_] = v

            self.load_state_dict(state_dict_new)

            print(f'Model successfully loaded from {path}\n')

    def _generate_from_prompt(self, inputs: Dict[str, Tensor]) -> List[str]:
        """
        Generates prediction from input prompt.

        :param inputs: prompt text [B, L]
        :return: predicted labels [B]
        """
        def decode(_ids) -> List[str]:
            return self.tokenizer.batch_decode(_ids, skip_special_tokens=True)

        out_b = self.model.generate(**inputs, max_new_tokens=2)
        out_b = decode(out_b)

        return out_b

    def forward(self, inputs: Dict[str, Tensor], labels: Tensor = None) -> Union[Tensor, List[str]]:
        # eval
        if labels is None:
            out = self._generate_from_prompt(inputs)
            return out

        # train
        else:
            inputs['labels'] = labels

            out = self.model(**inputs)  # [B, L, D]
            return out.loss

    @torch.inference_mode()
    def predict(self, text: str, top_k=None) -> List[str]:
        tokens = self.tokenizer(text, return_tensors="pt")
        tokens = tokens.input_ids.to(self.device)

        pred = self.model.generate(input_ids=tokens,
                                   max_length=4)

        pred = self.tokenizer.decode(token_ids=pred[0],
                                     skip_special_tokens=True)

        return pred


class ZeroShotLM(nn.Module):
    """
    Implements models supported under `pipeline`.
    """
    def __init__(self, task: str, name: str, device='cpu'):
        super().__init__()
        device = torch.device(device)

        self.model = pipeline(task=task,
                              model=name,
                              device=device)

        if task == 'fill-mask':
            self.t2t = False
            self.key = 'token_str'
        else:
            self.t2t = True
            self.key = 'generated_text'

        self.name = name

    def forward(self, text: str) -> str:
        if self.t2t:
            text = dict(text_inputs=text,
                        max_new_tokens=2)
        else:
            text = dict(inputs=text)

        pred = self.model(**text)

        pred = pred[0][self.key]

        return pred


if transformers.__version__ == '4.18.0.dev0':
    class OFA(nn.Module):
        """
        Implements OFA model for VQA (color extraction)
        """
        size = 384
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        # Monkey patch
        OFAModel.padding_idx = 1

        def __init__(self, path, device='cpu'):
            super().__init__()

            # Tokenizer
            self.tokenizer = OFATokenizer.from_pretrained(path)

            # Model
            self.model = OFAForConditionalGeneration.from_pretrained(path)
            self.model.eval()

            # Processor
            self.transform = Compose([lambda im: im.convert("RGB"),
                                      Resize((self.size, self.size), Image.BICUBIC),
                                      ToTensor(), Normalize(self.mean, self.std)])
            # Device
            self.to(device)

            self.device = device

        @torch.inference_mode()
        def forward(self, text: str, image: Image) -> str:
            # Preprocess
            text = self.tokenizer([text], return_tensors="pt")["input_ids"].to(self.device)
            image = self.transform(image).unsqueeze(0).to(self.device)

            # Generate
            out = self.model.generate(inputs=text,
                                      patch_images=image,
                                      num_beams=4,
                                      max_length=8)
            # Decode
            out = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            out = out[0].strip()

            return out

        @torch.inference_mode()
        def prompt(self, text: str) -> str:
            """
            Performs text-only prompting by masking out image.
            """
            # Preprocess
            text = self.tokenizer([text], return_tensors="pt")["input_ids"].to(self.device)
            image = torch.zeros([1, 3, self.size, self.size]).to(self.device)
            p_mask = torch.tensor([False]).to(self.device)

            # Generate
            out = self.model.generate(inputs=text,
                                      patch_images=image,
                                      patch_masks=p_mask,
                                      max_length=2)
            # Decode
            out = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            out = out[0].strip()

            return out

        @torch.inference_mode()
        def inference(self, batch_text: List[str], batch_image: List) -> List[str]:
            """
            Implements `forward()` at batch-level.
            """
            # Preprocess
            text = self.tokenizer(text=batch_text,
                                  padding=True,
                                  return_tensors='pt')
            text = text['input_ids']

            image = [self.transform(im) for im in batch_image]
            image = torch.stack(image)

            p_mask = torch.tensor([True] * len(image))

            # To Device
            text = text.to(self.device)
            image = image.to(self.device)
            p_mask = p_mask.to(self.device)

            # Generate
            out = self.model.generate(inputs=text,
                                      patch_images=image,
                                      patch_masks=p_mask,
                                      num_beams=4,
                                      max_length=8)

            # Decode
            out = self.tokenizer.batch_decode(out, skip_special_tokens=True)
            out = [o.strip() for o in out]

            return out


# -----------------------------------------
# ************ PIPELINE MODELS ************
# -----------------------------------------


class CLIPImageText(nn.Module):
    """
    Implements Image-to-Text Semantic Similarity Transformer
    """
    def __init__(self, device):
        super().__init__()
        self.model = SentenceTransformer('clip-ViT-L-14')
        self.device = device
        self.to(device)

    def forward(self, **kwargs):
        return self.inference(**kwargs)

    @torch.inference_mode()
    def inference(self,
                  image: Image,
                  anchor: str,
                  candidates: List[str],
                  pick_best=False) -> Dict[str, float]:
        """
        Selects `candidates` whose similarity score (dot)
        with query `image` exceeds that of the `anchor`.
        """
        # Prepare
        image = image.convert('RGB')

        # Image embeddings (normalized)
        image_emb = self.model.encode(image,
                                      normalize_embeddings=True,
                                      device=self.device,
                                      batch_size=1)
        # Pack
        text = [anchor] + candidates

        # Text embeddings (normalized)
        text_emb = self.model.encode(text,
                                     normalize_embeddings=True,
                                     device=self.device,
                                     batch_size=32)
        # Unpack
        anchor_emb = text_emb[0, :]
        candidates_emb = text_emb[1:, :]

        # Similarity (dot)
        scores = candidates_emb @ image_emb

        # Select Best
        if pick_best:
            i = scores.argmax()
            best = {candidates[i]: scores[i]}

            return best

        # Filter candidates scoring better than anchor
        preds = {c: score for c, score in zip(candidates, scores)}

        anchor_score = preds[anchor]

        tol = 1e-4
        selected = {c: score for c, score in preds.items()
                    if anchor_score <= score + tol}

        return selected


class TextSim(nn.Module):
    """
    Implements Semantic Similarity Model using Sentence-Transformer.
    """
    def __init__(self, device):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.device = device
        self.to(device)

    def forward(self, **kwargs):
        return self.inference(**kwargs)

    @torch.inference_mode()
    def inference(self, query: str, candidates: List[str], show=False) -> Union[str, Dict[str, float]]:
        """
        Selects best candidate, given the query.
        Optionally, returns candidates with scores.
        """
        # Compute query & candidate embeddings (normalized)
        embeddings = self.model.encode(sentences=[query] + candidates,
                                       normalize_embeddings=True,
                                       device=self.device,
                                       batch_size=32)
        q_emb = embeddings[0, :]
        c_emb = embeddings[1:, :]

        # Compute similarity (dot)
        scores = c_emb @ q_emb

        if show:
            # candidate scores
            scores = {c: score for c, score in zip(candidates, scores)}
            scores = sort_dict(scores, by='v', reverse=True)

            return scores

        else:
            # best candidate
            _i = scores.argmax()
            best = candidates[_i]

            return best


class DPTDepth(nn.Module):
    """
    Implements DPT Model for monocular depth estimation.
    """
    def __init__(self, device='cpu'):
        super().__init__()

        self.feat_ext = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.device = device

        self.to(device)

    @torch.inference_mode()
    def forward(self, image: Image) -> Image:
        # prepare image for the model
        inputs = self.feat_ext(images=image, return_tensors="pt")

        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        pred = outputs.predicted_depth

        # interpolate to original size
        prediction = F.interpolate(input=pred.unsqueeze(1),
                                   size=image.size[::-1],
                                   mode="bicubic")

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        return depth


class DPTSemSeg(nn.Module):
    """
    Implements DPT Model for semantic segmentation.
    """
    def __init__(self, device='cpu'):
        super().__init__()

        self.feat_ext = DPTFeatureExtractor.from_pretrained("Intel/dpt-large-ade")
        self.model = DPTForSemanticSegmentation.from_pretrained("Intel/dpt-large-ade")

        self.id2label = self.model.config.id2label
        self.label2id = {l: i for i, l in enumerate(self.id2label)}
        self.device = device

        self.to(device)

    @torch.inference_mode()
    def forward(self, image: Image) -> torch.LongTensor:
        # prepare image for the model
        inputs = self.feat_ext(images=image, return_tensors="pt")

        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        logits = outputs.logits.cpu()

        # pixel-wise class labels
        pred = torch.nn.functional.interpolate(input=logits,
                                               size=image.size[::-1],
                                               mode='bilinear')

        pred = pred.argmax(dim=1)[0]

        return pred


class UniCLImageText(nn.Module):
    """
    Implements UniCL model for Image-Text Similarity.
    """
    _DIR = __file__.split('/')[:-1]
    _DIR = '/'.join(_DIR) + '/unicl'

    cfg_path = osj(_DIR, 'configs/unicl_swin_base.yaml')
    ckpt_path = osj(_DIR, 'in21k_yfcc14m_gcc15m_swin_base.pth')

    def __init__(self, device: str, obj_emb_path: str = None, img_emb_path: str = None):
        super().__init__()
        # Config
        self.cfg = get_config(self.cfg_path)
        self.device = device

        # Precomputed Embeddings
        self.text_emb = read_pkl(obj_emb_path) if obj_emb_path else None
        self.img_emb = read_pkl(img_emb_path) if img_emb_path else None

        # Model
        self.model = UniCLModel(self.cfg)
        self._load_wts()
        self.model.eval()

        self.transform = self._build_transforms()
        self.to(device)

    def _load_wts(self):
        ckpt = torch.load(self.ckpt_path, 'cpu')

        self.model.load_state_dict(ckpt["model"])

    @staticmethod
    def _build_transforms(im_size=224, center_crop=True):
        trf = [T.Resize(im_size)]
        if center_crop:
            size = int((256 / 224) * im_size)
            trf = [T.Resize(size), T.CenterCrop(im_size)]
        trf += [T.ToTensor(), T.Normalize(M, S)]
        return T.Compose(trf)

    def forward(self, **kwargs):
        return self.inference(**kwargs)

    @torch.inference_mode()
    def inference(self,
                  image: Image,
                  anchor: str,
                  candidates: List[str],
                  pick_best: bool = False,
                  img_id: str = None) -> Dict[str, float]:
        """
        Selects `candidates` whose similarity score (dot)
        with query `image` exceeds that of the `anchor`.

        If `pick_best` is set, returns max scoring candidate.
        """
        # Image
        if img_id and self.img_emb:
            img_emb = self.img_emb[img_id].to(self.device)
        else:
            image = image.convert('RGB')
            image = self.transform(image).unsqueeze(0)

            img_emb = self.model.encode_image(image.to(self.device))

        # Text
        if self.text_emb:
            text_emb = torch.stack([self.text_emb[o] for o in candidates]).to(self.device)
        else:
            text_emb = self.model.get_text_embeddings(candidates, self.device)

        # Score
        output = self.model.logit_scale.exp() * img_emb @ text_emb.t()
        scores = output.softmax(-1).flatten()

        # Select Best
        if pick_best:
            i = scores.argmax()
            best = {candidates[i]: scores[i].item()}

            return best

        # Filter candidates scoring better than anchor
        preds = {c: score.item() for c, score in zip(candidates, scores)}

        anchor_score = preds[anchor]

        tol = 1e-4
        selected = {c: score for c, score in preds.items()
                    if anchor_score <= score + tol}

        return selected

    @torch.inference_mode()
    def precompute_text_emb(self, objs: List[str], save_fp: str):
        obj2emb = {}
        for o in tqdm(objs):
            emb = self.model.get_text_embeddings([o], self.device)[0]
            obj2emb[o] = emb.cpu()

        save_pkl(obj2emb, save_fp)

    @torch.inference_mode()
    def precompute_image_emb(self, im_paths: List[str], save_fp: str):
        from color.constants import IGNORE_IMAGES

        img2emb = {}
        for path in tqdm(im_paths):
            im_id = path.split('/')[-1].split('.jpg')[0]

            if int(im_id) not in IGNORE_IMAGES:
                im = Image.open(path).convert('RGB')
                im = self.transform(im).unsqueeze(0)

                emb = self.model.encode_image(im.to(self.device))[0]
                img2emb[im_id] = emb.cpu()

        save_pkl(img2emb, save_fp)


class POSTagger(nn.Module):
    """
    Implements Part-of-Speech Tagger (BiLSTM-CRF)
    """

    def __init__(self):
        super().__init__()
        self.tagger = SequenceTagger.load("flair/pos-english")

    def forward(self, sent: str) -> Dict[str, str]:
        # Preprocess
        words = sent.split()
        sent = Sentence(sent)

        # Tagging
        self.tagger.predict(sent)
        tags = [word.value for word in sent.labels]

        # Dict
        token_tag_dict = {word: tag for word, tag in zip(words, tags)}

        return token_tag_dict


def _demo():
    def _plot(_im: Image, save: str = None):
        plt.figure(figsize=(14, 14))
        plt.imshow(_im, cmap='gray')
        if save:
            plt.savefig(save, bbox_inches='tight')
        else:
            plt.show()

    im = None
    txt = "what color is the mirror?"

    # Color
    ofa = OFA(path=os.environ['OFA'], device='cuda:0')
    _o = ofa(txt, im)
    print(_o)

    o_b = ofa.inference([txt, txt], [im, im])
    print(o_b)

    # m = DepthTransformer('cpu')
    # o = m(im); _plot(o)

    # import requests
    # from glob import glob
    # from utils import Timer
    #
    # url = 'https://t4.ftcdn.net/jpg/02/99/23/49/360_F_299234924_cBWFTruM7TGGoahcDabUn0b65PTtiZOA.jpg'
    # img = Image.open(requests.get(url, stream=True).raw)
    #
    # obj = 'ball'
    # cds = read_json('./data/object_subtypes_o100_s10.json')

    # im2txt = UniCLImageText('cuda:1', './data/obj_emb.pkl', './data/img_emb.pkl')
    # t = Timer('Precomputed')
    # for _ in range(50):
    #     im2txt.inference(None, obj, list(cds[obj]), img_id='1')
    # t.end()

    # im2txt = UniCLImageText('cuda:1')

    # Image
    # img_paths = glob('./VG/images_1/*.jpg') + glob('./VG/images_2/*.jpg')
    # im2txt.precompute_image_emb(img_paths, save_fp='./data/img_emb.pkl')

    # Text
    # objects = list(read_json('./data/object_names_c5.json'))
    # im2txt.precompute_text_emb(objects, save_fp='./data/obj_emb.pkl')


def _depth_ade20k():
    _dir = '/home/axe/Datasets/ADE20K_2021_17_01'

    paths = glob(osj(_dir, '**', '*.jpg'), recursive=True)

    model = DPTDepth(device='cuda:0')

    for p in tqdm(paths):
        # Image
        img = Image.open(p).convert('RGB')

        # Depth
        depth = model(img)

        # Save
        depth.save(p.replace('.jpg', '_depth.jpeg'))


def _depth_vg():
    from color.constants import IGNORE_IMAGES

    def _path2id(path: str):
        path = path.split('/')[-1]
        path = path.split('.')[0]
        return int(path)

    _dir = '/home/axe/Datasets/Visual_Genome'

    paths = glob(osj(_dir, '**', '*.jpg'), recursive=True)

    model = DPTDepth(device='cuda:1')

    for p in tqdm(paths):
        # Skip corrupt images
        _id = _path2id(p)
        if _id in IGNORE_IMAGES:
            continue

        # Image
        img = Image.open(p).convert('RGB')

        # Depth
        depth = model(img)

        # Save
        p = p.replace('images', 'depth')

        depth.save(p)


def _prompt():
    # m = TodoLM(task='fill-mask', name='bert-base-uncased')
    # prompt = "banana is of [MASK] color"
    # m = TodoLM(task='text-generation', name='gpt2', device='cuda:1')
    # prompt = "the color of banana is "
    # m = UnifiedQA(name='allenai/unifiedqa-t5-small', device='cuda:1')
    # prompt = "is banana red in color?"
    m = ZeroShotLM(task='fill-mask', name='uclanlp/visualbert-vqa-coco-pre')

    print(m.predict("apple is of [MASK] color", top_k=5))


if __name__ == '__main__':
    _d = 'cpu'
    _B, _L, _C = 4, 16, 3

    _inp = dict(input_ids=torch.randint(100, [_B, _L]).to(_d),
                attention_mask=torch.ones([_B, _L]).to(_d))
    _lbl = torch.randint(_C, [_B])

    _inp['pixel_values'] = torch.rand([_B, 3, 224, 224]).to(_d)

    m = MaskedLM('facebook/flava-full', _C, _d, vl=True)

    # def _to_tensor(data: dict):
    #     return {k: torch.tensor(v) for k, v in data.items()}
    #
    # m = UnifiedQA('allenai/unifiedqa-t5-small', device=d)
    # inp = m.tokenizer(["color of grass is", "color of tomato is"])
    # lbl = m.tokenizer(['green', 'red'])
    #
    # inp = _to_tensor(inp.data)
    # lbl = torch.tensor(lbl.input_ids)

    _o = m(_inp, _lbl)
    print(_o)
