"""
Stores parameters for building & cleaning the dataset development pipeline.
"""

# VG Images (corrupt IDs)
IGNORE_IMAGES = [2331541, 2416218, 2325661, 2330142, 2403972, 2400491,
                 2391568, 2326141, 2316472, 2369516, 2378996, 2415090,
                 2321571, 2350190, 2337155, 2404941, 2336131, 2417802,
                 2392406, 2363899, 2335991, 2340698, 2320098, 2378352,
                 2347067, 2395307, 2336852, 2372979, 2402595, 2336771,
                 2357002, 2350687, 2345238, 2405946, 2382669, 2394180,
                 2392550, 2392657, 2401212, 2328613, 2395669, 2328471,
                 2380334, 2402944, 2373542, 2394503, 2354486, 2393276,
                 2333919, 2410274, 2393186, 2415989, 2408825, 2386339,
                 2407337, 2414917, 2354293, 2408461, 2387773, 2335434,
                 2343497, 2388568, 2379781, 2387074, 2370307, 2339625,
                 2388085, 2414491, 2417886, 2401793, 2354437, 2388917,
                 2383685, 2327203, 2346773, 2390387, 2351122, 2397804,
                 2385744, 2369047, 2412350, 2346745, 2323189, 2316003,
                 2339038, 2344194, 2389915, 2406223, 2329481, 2370916,
                 2350125, 2386269, 2368946, 2414809, 2323362, 2396395,
                 2410589, 2405591, 2357389, 2370544, 2407333, 2396732,
                 2339452, 2360452, 2337005, 2329477, 2400248, 2408234,
                 2335721, 2348472, 2319788, 2319311, 2380814, 2374336,
                 2356194, 2379468, 2318215, 2335839, 2343246, 2384303,
                 2338403, 2345400, 2360334, 2397721, 2376553, 2346562,
                 2325435, 2390322, 2370333, 2320630, 2416204, 2370885,
                 2321051, 2405004, 2400944, 2379210, 2390058, 2374775,
                 2405254, 2388989, 2356717, 2416776, 2413483, 2344903,
                 2369711, 2357151, 2344765, 2338884, 2357595, 2359672,
                 2385366, 2361256, 2379927, 2407778, 2344844, 2340300,
                 2315674, 2394940, 2325625, 2355831, 2324980, 2388284,
                 2399366, 2407937, 2354396, 2397424, 2386421]


# Threshold values
OBJECT_MIN_COUNT = 5
OBJ_SUB_MIN_COUNT = 100
SUBTYPE_MIN_COUNT = 10


# Map Attribute to Color
ATTR2COLOR = {'gold': 'yellow', 'golden': 'yellow', 'blond': 'yellow', 'blonde': 'yellow',
              'wooden': 'brown', 'tan': 'brown', 'beige': 'yellow brown', 'bronze': 'brown',
              'grey': 'gray', 'silver': 'gray', 'metal': 'gray', 'steel': 'gray', 'copper': 'brown',
              'peach': 'yellow pink', 'cream': 'yellow', 'violet': 'purple',
              'maroon': 'red', 'turquoise': 'blue', 'teal': 'blue green'}

# Primary Colors
COLOR_SET = ['red', 'orange', 'yellow', 'brown', 'green',
             'blue', 'purple', 'pink', 'white', 'gray', 'black']

# Colors mentioned in VG
ALL_COLORS = COLOR_SET + ['grey', 'violet', 'gold', 'golden', 'blond', 'blonde', 'tan', 'beige',
                          'bronze', 'silver', 'peach', 'cream', 'maroon', 'turquoise', 'teal']


# Ignore objects (color)
IGNORE_OBJECTS = ['person', 'man', 'woman', 'lady', 'boy', 'girl', 'child', 'adult', 'someone', 'part', 'piece', 'edge',
                  'side', 'background', 'scene', 'distance', 'shadow', 'air', 'pair', 'color', 'colour', 'thing', 'guy',
                  'word', 'room', 'silver']

IGNORE_OBJECTS += ['door refrigerator', 'door car']


# Drop object qualifiers; See `_drop_objs_and_merge()`
DROP_QUALIFIERS = ['color', 'colour', 'word', 'gren', 'blu', 'whtie', 'otuside', 'buildinga',
                   'center', 'side', 'front', 'top', 'bottom', 'right', 'left', 'corner',
                   'piece', 'style', 'background', 'metal', 'wood', 'steel']

DROP_QUALIFIERS += ['dell', 'mac', 'nike', 'coca cola', 'adidas']


# Include objects
ADD_OBJECTS = ['wall', 'umbrella', 'cd', 'grizzly bear', 'racing suit', 'walk sign', 'drawer', 'handle', 'switch', 'ox',
               'sink', 'stand', 'railing', 'plastic container', 'plastic bottle', 'frying pan', 'ceiling light', 'suv',
               'recycling bin', 'can', 'chalk', 'sun glass', 'stop sign', 'dress', 'hat', 'plastic lid', 'sleeve shirt',
               'balcony railing', 'mug', 'remote control', 'remote controller', 'cooking utensil', 'rug', 'silverware',
               'tv set', 'tv remote', 'flourescent light', 'awning', 'kayak', 'bus stop sign', 'shop sign', 'olive',
               'trash can', 'handle grip', 'faucet handle', 'ear', 'stove top', 'stove hood', 'stop light', 'oven hood',
               'pen', 'mat', 'vent', 'tie', 'folding chair', 'yard', 'zebra', 'eye', 'building roof', 'garbage can',
               'bare tree', 'leafless tree', 'grease spot', 'warning sign', 'slice', 'spray bottle', 'bus stop', 'bra',
               'spoke', 'knee pad', 'ceiling fan', 'railing fence', 'oven door', 'oatmeal', 'reclining chair', 'grease',
               'steering wheel', 'gecko', 'flip flop', 'exhaust pipe', 'cross walk', 'bow tie', 'soda can', 'bandana',
               'smoke', 'tee shirt', 'beer', 'rose', 'flush lever', 'bow', 'mesh', 'ac unit', 'shovel', 'net', 'elk',
               'jean short', 'bell', 'cereal box', 'police man', 'dough', 'deer', 'gown', 'peg', 'oven burner', 'nacho',
               'bowling pin', 'bowling lane', 'bowling shoe', 'bowling ball', 'cleat', 'denim short', 'exhaust fan',
               'ceiling lamp', 'turntable', 'bathing suit', 'buckle', 'post-it note', 'oven handle', 'crease', 'keg',
               'shaving cream', 'stop signal', 'gum', 'hiking path', 'pelican', 'us flag', 'cole slaw', 'wig', 'die',
               'flip phone', 'hoody', 'stop button', 'barb wire', 'camping tent', 'beer can', 'oven window', 'bunny',
               'stormy sky', 'semi truck', 'brussel sprout', 'gem', 'ceiling lighting', 'key chain', 'underwear', 'ham',
               'kitty', 'touring bus', 'rowing boat', 'usb', 'pug', 'pug dog', 'knit cap', 'rum', 'wet suit', 'bear',
               'light house', 'smart phone', 'wii remote', 'flying disc', 'lighter', 'dough nut', 'oven rack', 'watch',
               'ear piece', 'sailing ship', 'ankle brace', 'fireman hat', 'flip cellphone', 'spandex short', 'bear cub',
               'visor hat', 'camping bag', 'police sign', 'monk robe', 'oven top', 'police hat', 'avocado slice', 'yak',
               'knitting needle', 'scissor handle', 'baking oven', 'grizzly', 'nintendo wii remote', 'mane hair', 'rat',
               'teddy bear nose', 'flying kite', 'slice bread', 'bear bottle', 'dye', 'zipper handle', 'flush tank',
               'fry pan', 'knit', 'crank', 'spatula handle', 'brake handle', 'mug handle', 'wheel spoke', 'foam box',
               'grizzly bear', 'pepperoni topping', 'ac vent', 'window ac', 'leg', 'jetski', 'tv', 'butterknife',
               'mousepad', 'faceguard', 'cellotape', 't-shirt', 'tee-shirt', 'sweatpant', 'lofa', 'dvd', 'tanktop',
               'touchpad', 'paperbag', 'soccerball', 'lightpost', 'trashbin', 'merry-go-round', 'tshirt', 'blind',
               'knife handle']

ADD_OBJECTS += ['tv stand', 'broccoli floret', 'tv screen', 'double-decker bus', 'doubledecker bus', 'dvd player',
                'chain-link fence', 'post-it note', 'pick-up truck', 'flatscreen tv', 'mp3 player', 'cd case', 'cd-rom',
                'dvd case', 'lcd screen', 'tv monitor', 'tv tray', 'tv camera', 'multistory building', 'pvc pipe',
                'u-turn sign', 'cd player', 'apple macbook', 'macbook pro', 'highrise building', 'crt monitor',
                'sideview mirror', 'tv cabinet', 'bbq sauce', 'rear-view mirror', 'afternoon sky', 'cd drive']


# Ignore Size
SKIP_TERMS = ['wall', 'floor', 'sky', 'cloud', 'area', 'road', 'pavement', 'street', 'sidewalk', 'sand', 'street',
              'sea', 'ocean', 'lake', 'river', 'water', 'ground', 'grass', 'edge', 'top', 'hair', 'part', 'concrete',
              'picture', 'photo', 'corner', 'line', 'shadow', 'reflection', 'item', 'color', 'mark', 'view', 'grove',
              'circle', 'par', 'core', 'construction', 'work', 'groove', 'crossing', 'zoo', 'shadow cast', 'engineer',
              'land', 'dad', 'grout', 'shirt boy', 'mountainside', 'porcelain', 'focus', 'system', 'text', 'toiletry',
              'structure', 'flop', 'stuff', 'layer', 'angel', 'font', 'steam', 'curve', 'product', 'competition', 'sun',
              'half', 'support', 'cattle grazing', 'balance scale', 'material', 'profile', 'iii', 'bedding', 'bloat',
              'empire', 'trio', 'price', 'route', 'flooring', 'statute', 'trick', 'caramel', 'group', 'breakfast',
              'nick knack', 'image', 'blur', 'chunk', 'algae', 'gravy', 'soil', 'destination', 'water side', 'runner',
              'art', 'roman', 'season', 'horizon', 'team', 'direction', 'elephant feeding', 'world war', 'dinosaur',
              'ray', 'maroon', 'mother', 'print', 'food', 'section', 'movie', 'metal', 'cityscape', 'batch', 'birthday',
              'town', 'xii', 'expression', 'piece', 'habitat', 'word', 'city', 'photographer', 'fall', 'place', 'set',
              'alphabet', 'scene', 'instruction', 'fun', 'river side', 'car driving', 'game', 'side', 'video', 'front',
              'position', 'capital', 'business', 'character', 'lot', 'sun', 'moon']


# Remove `main`
if __name__ == '__main__':
    from spellchecker import SpellChecker
    from utils import read_json

    jsn = read_json('../data/object_names_c5.json')
    multi_words = [(o, f) for o, f in jsn.items() if len(o.split()) in [2, 3]]

    spell = SpellChecker()
    for multi_word, freq in multi_words:
        typos = spell.unknown(multi_word.split())
        for typo in typos:
            correction = spell.correction(typo)
            print(multi_word, ':', typo, '-->', correction, '   |   ', freq)
