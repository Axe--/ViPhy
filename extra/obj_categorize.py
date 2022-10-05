"""
Assigns object to a category (predefined set).
"""
from tqdm import tqdm
from models import TextSim
from utils import read_json, save_json

CATEGORIES = \
"""
human body: hand, fist, mouth, lips, beard, hair, teeth, skin
animal body: paw, hooves, tail, mane, fur, whisker, flesh
person: man, woman, lady, boy, girl, child, baby
animal: bird, fish, rodent, reptile, insect, pet, cattle, mammal, carnivore
nature: sky, tree, branch, flower, forest, farm, field, sea, lake, desert, beach, mountain, snow, smoke
food: vegetable, fruit, dessert, beverage, snack, fast food
furniture: table, chair, shelf, bed, cabinet
kitchenware: oven, fridge, knife, bowl, kitchen sink
bathroom: towel, shower, toothbrush, bathroom sink 
sports equipment: helmet, racket, ball, glove, skateboard
clothing: hat, shirt, pant, skirt, footwear, shoe, boot, sunglasses, canopy, tent, umbrella
electronics: laptop, mouse, keyboard, printer, socket, phone, speaker, projector, game
vehicle: bike, car, bus, truck, boat, plane
street: traffic, light, road sign, crosswalk, sidewalk, hydrant, pole
construction: bridge, building, house, hospital, school, hotel, factory, airport, stadium, store
utility items: lock key, lamp, paper, pen, stationery, whiteboard, trashcan, cart, bag, extinguisher
miscellaneous: entity, object, thing, stuff, toy, unknown, unidentified
"""
# music instrument: guitar, piano, keyboard, flute, drum

if __name__ == '__main__':
    # Args
    save = './obj2cat.json'
    model = TextSim('cuda:1')

    objects = read_json('../results/colors.json')

    candidates = CATEGORIES.strip('\n').split('\n')

    obj2cat = {}
    for obj in tqdm(objects):
        out = model.inference(obj, candidates)

        if save:
            obj2cat[obj] = out.split(':')[0]
        else:
            out = {c.split(':')[0]: int(score*100) for c, score in out.items()}
            print(obj, '\n', out, '\n\n')

    save_json(obj2cat, path=save)
