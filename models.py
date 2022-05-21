import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Dict, List
from flair.data import Sentence
from flair.models import SequenceTagger

import transformers

if transformers.__version__ == '4.18.0':
    from transformers import DPTFeatureExtractor, DPTForDepthEstimation
elif transformers.__version__ == '4.18.0.dev0':
    from transformers import OFATokenizer, OFAForConditionalGeneration

from torchvision.transforms import Compose, Resize, ToTensor, Normalize


class POSTagger(nn.Module):
    """
    Implements Part-of-Speech Tagger (LSTM-CRF)
    """
    def __init__(self):
        super().__init__()
        self.tagger = SequenceTagger.load("flair/pos-english")

    def forward(self, sent: str) -> List[Dict]:
        tokens = sent.split()

        sent = Sentence(sent)
        # Tagging
        self.tagger.predict(sent)

        tags = [word.value for word in sent.labels]

        token_tag_list = [dict(token=tok, tag=tag) for tok, tag in zip(tokens, tags)]

        return token_tag_list


class OFA(nn.Module):
    """
    Implements OFA model for VQA (color extraction)
    """
    size = 384
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    def __init__(self, path, device='cpu'):
        super().__init__()

        # Tokenizer
        self.tokenizer = OFATokenizer.from_pretrained(path)

        # Model
        self.model = OFAForConditionalGeneration.from_pretrained(path)

        # Processor
        self.transform = Compose([lambda im: im.convert("RGB"),
                                  Resize((self.size, self.size), Image.BICUBIC),
                                  ToTensor(), Normalize(self.mean, self.std)])
        # Device
        self.to(device)

    def forward(self, text: str, image: Image) -> str:
        # Process
        text = self.tokenizer([text], return_tensors="pt")["input_ids"]
        image = self.transform(image).unsqueeze(0)

        # Generate
        out = self.model.generate(inputs=text,
                                  patch_images=image,
                                  num_beams=4,
                                  max_length=8)
        # Decode
        out = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        out = out[0].strip()

        return out


class DepthTransformer(nn.Module):
    """
    Implements Depth prediction model (Transformer)
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
                                   size=image.im_size[::-1],
                                   mode="bicubic")

        # visualize the prediction
        output = prediction.squeeze().cpu().numpy()
        formatted = (output * 255 / np.max(output)).astype("uint8")
        depth = Image.fromarray(formatted)

        return depth


def _plot(im: Image):
    # plt.figure(figsize=(14, 14))
    plt.imshow(im)
    plt.show()


# class SpellCorrector(nn.Module):
#     def __init__(self):
#         super().__init__()
#         from textblob import TextBlob
#         self.corrector = TextBlob
#
#     def forward(self, sent: str) -> str:
#         return ' '.join([self.corrector(word) for word in sent.split()])


if __name__ == '__main__':
    import requests

    url = 'https://image.cnbcfm.com/api/v1/image/107037293-IMG-9997.jpg?v=1648561728'
    img = Image.open(requests.get(url, stream=True).raw)

    txt = "what color is the mirror?"

    # Color
    model = OFA(path='../OFA/OFA-large')
    o = model(txt, img)

    print(o)
    _plot(img)

    # Depth
    # model = DepthTransformer(device='cuda:0')
    # o = model(img)
    # _plot(o)
