import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List
from flair.data import Sentence
from flair.models import SequenceTagger
from sentence_transformers import SentenceTransformer

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


class Image2TextSimilarity(nn.Module):
    """
    Implements Image-to-Text Semantic Similarity Transformer
    """
    def __init__(self, device):
        super().__init__()
        self.model = SentenceTransformer('clip-ViT-L-14')
        self.device = device

    def forward(self, **kwargs):
        return self.inference(**kwargs)

    @torch.inference_mode()
    def inference(self, img_path: str, anchor: str, candidates: List[str], top_k: int = None) -> List[str]:
        """
        Selects `candidates` whose similarity score (dot) with the query image
        exceeds that of the `anchor`.
        Returns the `top-k` candidates, if provided.
        """
        # Load Image
        image = Image.open(img_path).convert('RGB')

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
        candidate_scores = candidates_emb @ image_emb
        anchor_score = anchor_emb @ image_emb

        # Select candidates that score better than anchor
        idxs = (anchor_score <= candidate_scores)

        selected = np.asarray(candidates)[idxs].tolist()

        # Filter top-k
        selected = selected[:top_k] if top_k else selected

        return selected


class Text2TextSimilarity(nn.Module):
    """
    Implements Text-to-Text Semantic Similarity Transformer.
    """
    def __init__(self, device):
        super().__init__()
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.device = device

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


def _plot(img: Image):
    # plt.figure(figsize=(14, 14))
    plt.imshow(img)
    plt.show()


def _test():
    import requests

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

    # Depth
    # model = DepthTransformer(device='cuda:0')
    # o = model(im)
    # _plot(o)

    # Similarity
    # model = Text2TextSimilarity(device='cpu')
    #
    # res = model.inference(query='person at the hospital bed',
    # candidates=['glove', 'medical glove', 'baseball glove', 'ski glove'])
    # print(res)


if __name__ == '__main__':
    import requests

    im_path = './VG/old-fridge.jpg'
    cds = ['handle', 'door handle', 'knife handle', 'drawer handle', 'umbrella handle',
           'car door handle', 'bike handle', 'luggage handle', 'fridge handle', 'toilet handle']

    im2txt = Image2TextSimilarity('cpu')

    res = im2txt.inference(im_path, cds[0], cds)
    print(res)
