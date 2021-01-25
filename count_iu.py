import json

iu = json.load(open("..//iu_xray//dataset_iu.json","r"))

count = {"train":0,"test":0,"val":0}
for img in iu["images"]:
    count[img["split"]] += 1
print(count)
