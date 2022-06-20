"""
Compare Image-Text models
"""

if __name__ == '__main__':
    import requests
    from PIL import Image
    from utils import read_json, sort_dict
    from models import CLIPImageText, UniCLImageText
    from PIL.Image import UnidentifiedImageError

    subs = read_json('../data/object_subtypes_o100_s10.json')

    clip = CLIPImageText('cuda:1')
    unicl = UniCLImageText('cuda:1')

    while True:
        url = input('Enter URL:')
        new = input('Object:')

        if new != '':
            OBJ = new

        if url == 'q':
            break

        try:
            im = Image.open(requests.get(url, stream=True).raw)
        except UnidentifiedImageError:
            print("Image cannot be loaded. Try different URL!")
            continue

        if 1:
            print('CLIP')
            res = clip.inference(im, OBJ, list(subs[OBJ]))
            res = sort_dict(res, by='v', reverse=True)
            print(res)

        if 1:
            print('UniCL')
            res = unicl.inference(im, OBJ, list(subs[OBJ]))
            res = sort_dict(res, by='v', reverse=True)
            print(res)

        print('\n---------------------------------------------\n')
