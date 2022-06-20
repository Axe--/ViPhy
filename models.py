import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from os.path import join as osj
from typing import Dict, List, Union
from flair.data import Sentence
from flair.models import SequenceTagger
from sentence_transformers import SentenceTransformer

from torchvision import transforms as T
from timm.data.constants import IMAGENET_DEFAULT_MEAN as M
from timm.data.constants import IMAGENET_DEFAULT_STD as S

from unicl.config import get_config
from unicl.model.model import UniCLModel

from tqdm import tqdm
from utils import read_pkl, read_json, save_pkl

import transformers

# DPT
if transformers.__version__ >= '4.19.0':
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# OFA
if transformers.__version__ == '4.18.0.dev0':
    from transformers import OFATokenizer, OFAForConditionalGeneration, OFAModel
    from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class POSTagger(nn.Module):
    """
    Implements Part-of-Speech Tagger (LSTM-CRF)
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

        # Processor
        self.transform = Compose([lambda img: img.convert("RGB"),
                                  Resize((self.size, self.size), Image.BICUBIC),
                                  ToTensor(), Normalize(self.mean, self.std)])
        # Device
        self.to(device)

        self.device = device

    @torch.inference_mode()
    def forward(self, text: str, image: Image) -> str:
        # Process
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
    def inference(self, batch_text: List[str], batch_image: List) -> List[str]:
        """
        Implements `forward()` at batch-level.
        """
        # Process
        text = self.tokenizer(text=batch_text,
                              padding=True,
                              return_tensors='pt')
        text = text['input_ids']

        image = [self.transform(img) for img in batch_image]
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
                  pick_best=False,
                  caption: str = None,
                  top_k: int = None) -> Dict[str, float]:
        """
        Selects `candidates` whose similarity score (dot)
        with query `image` exceeds that of the `anchor`.

        Returns the `top-k` candidates, if provided.

        TODO: include region `caption` as additional query
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


class Text2TextSimilarity(nn.Module):
    """
    Implements Text-to-Text Semantic Similarity Transformer.
    """
    def __init__(self, device):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.device = device
        self.to(device)

    def forward(self, **kwargs):
        return self.inference(**kwargs)

    @torch.inference_mode()
    def inference(self, query: str, candidates: List[str]) -> str:
        """
        Selects best candidate, given the query.
        """
        # Compute query & candidate embeddings (normalized)
        embeddings = self.model.encode(sentences=[query] + candidates,
                                       normalize_embeddings=True,
                                       device=self.device,
                                       batch_size=32)
        q_emb = embeddings[0, :]
        c_emb = embeddings[1:, :]

        # Compute similarity (dot)
        sim_scores = c_emb @ q_emb

        # Best candidate
        _i = sim_scores.argmax()
        best = candidates[_i]

        return best


class DepthTransformer(nn.Module):
    """
    Implements Monocular Depth Model
    """
    def __init__(self, device='cpu'):
        super().__init__()

        self.feat_ext = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        self.device = device

        self.to(device)

    @torch.inference_mode()
    def forward(self, image: Image) -> Image:       # TODO: Handle Batch of Images (list)
        # prepare image for the model
        inputs = self.feat_ext(images=image, return_tensors="pt")

        # move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        pred = outputs.predicted_depth

        # interpolate to original size
        prediction = F.interpolate(input=pred.unsqueeze(1),
                                   size=image.im_size[::-1],
                                   mode="bicubic")

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        return depth


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
            text_emb = torch.stack([self.text_emb[o] for o in candidates])
            text_emb = text_emb.to(self.device)
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
            obj2emb[o] = emb

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
                img2emb[im_id] = emb

        save_pkl(img2emb, save_fp)


def _test():
    def _plot(_im: Image):
        # plt.figure(figsize=(14, 14))
        plt.imshow(_im)
        plt.show()

    url = 'https://image.cnbcfm.com/api/v1/image/107037293-IMG-9997.jpg?v=1648561728'
    im = Image.open(requests.get(url, stream=True).raw)
    # _plot(im)

    txt = "what color is the mirror?"

    # Color
    model = OFA(path='../OFA/OFA-large', device='cuda:0')
    _o = model(txt, im)
    print(_o)

    o_b = model.inference([txt, txt], [im, im])
    print(o_b)

    # model = DepthTransformer(device='cuda:0')
    # o = model(im); _plot(o)


if __name__ == '__main__':
    import requests
    from glob import glob
    from utils import Timer

    url = 'https://t4.ftcdn.net/jpg/02/99/23/49/360_F_299234924_cBWFTruM7TGGoahcDabUn0b65PTtiZOA.jpg'
    img = Image.open(requests.get(url, stream=True).raw)

    obj = 'ball'
    cds = read_json('./data/object_subtypes_o100_s10.json')

    # im2txt = UniCLImageText('cuda:1', './data/obj_emb.pkl', './data/img_emb.pkl')

    # t = Timer('Precomputed')
    # for _ in range(50):
    #     im2txt.inference(None, obj, list(cds[obj]), img_id='1')
    # t.end()

    # t = Timer('UniCL')
    # for _ in range(50):
    #     res = im2txt.inference(img, obj, list(cds[obj]))
    # t.end()

    im2txt = UniCLImageText('cuda:1')
    # img_paths = glob('./VG/images_1/*.jpg') + glob('./VG/images_2/*.jpg')
    # im2txt.precompute_image_emb(img_paths, save_fp='./data/img_emb.pkl')

    objects = list(read_json('./data/object_names_c5.json'))
    im2txt.precompute_text_emb(objects, save_fp='./data/obj_emb.pkl')
