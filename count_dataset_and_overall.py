import json

# train_cnt = 0
# val_cnt = 0
# test_cnt = 0
# with open('data/dataset_mimic.json') as json_file:
#     data = json.load(json_file)
#     # print(data["images"])
#     for img in data["images"]:

#         if img['split']== 'train':
#             train_cnt = train_cnt + 1
#         elif img['split']== 'val':
#             val_cnt = val_cnt + 1
#         elif img['split']== 'test':
#             test_cnt = test_cnt + 1
#     print('train', train_cnt, 'val', val_cnt, 'test', test_cnt)

# with open('eval_results/mimic_tf_test_hid.json') as json_file:
#     data = json.load(json_file)
#     keys = data.keys()
#     print(data["overall"]["Bleu_1"])
#     imgs = data["imgToEval"]
#     print(len(imgs.keys()))
#     # print(imgs)
#     for im in data:
#         i = 0

# file_name = 'eval_results/a2i2_test.json'
# file_name = 'eval_results/fc_test.json'
file_name = 'eval_results/m2_test.json'
with open(file_name) as json_file:
    data = json.load(json_file)
    keys = data.keys()
    print(data["overall"])

