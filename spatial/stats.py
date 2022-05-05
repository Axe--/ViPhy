import pickle
from glob import glob
from os.path import join as osj


if __name__ == '__main__':
    # Load Dataset
    root_dir = '/home/axe/Datasets'

    with open(root_dir + '/ADE20K_2021_17_01/index_ade20k.pkl', 'rb') as f:
        ade20k = pickle.load(f)

    # scenes = [f.replace('/training', '').replace('/validation', '') for f in ade20k['folder']]
    # TODO: Also include `validation`

    scenes = [f for f in ade20k['folder'] if 'training' in f]

    scenes = sorted(list(set(scenes)))

    img_per_scene = {}

    for scene in scenes:
        path = osj(root_dir, scene, '*.jpg')
        imgs = glob(path)
        name = scene.split('training/')[1]

        img_per_scene[name] = len(imgs)

    print(sum(1 for x in img_per_scene.values()))
    print(len(img_per_scene))

    img_per_scene = sorted(img_per_scene.items(), key=lambda x: x[1], reverse=True)
    print(img_per_scene[:100])
