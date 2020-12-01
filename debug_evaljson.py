import json
with open('data/mimic_eval.json') as json_file:
    data = json.load(json_file)
    print(data.keys())
    images = data["images"]
    print(images[20])
    print(len(images))
    print()