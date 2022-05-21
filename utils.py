import json


# ------------------------------------------------
def read_json(path):
    with open(path) as json_data:
        data = json.load(json_data)
    return data


def save_json(data, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
