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
# file_name = 'eval_results/m2_test.json'
# file_name = 'data/dataset_mimic.json'
file_name = 'data/mimic_eval.json'
with open(file_name) as json_file:
    mimic_dataset = json.load(json_file)
    keys = mimic_dataset.keys()
    print('keys', keys)
print('info', mimic_dataset['info'])
print('licenses', mimic_dataset['licenses'])
print('type', mimic_dataset['type'])
print('dataset', mimic_dataset['dataset'])
print('image length', len(mimic_dataset['images']))
print('annotations length', len(mimic_dataset['annotations']))
# print(mimic_dataset['images'][1])
# print(mimic_dataset['annotations'][10])
# print(data["overall"])
#     print(keys)
#     print(data['dataset'])
# print(mimic_dataset['images'][0].keys())
# print(mimic_dataset['images'][100])
retina_dataset = {}
retina_dataset['info'] = 'Not avilable'
retina_dataset['licenses'] = 'No licenses'
retina_dataset['type'] = 'caption'
retina_dataset['dataset'] = 'mimic'
# load the dataset using dataset_retina_resize.json
retina_file_name = 'data/dataset_retina_resize.json'
with open(retina_file_name) as retina_json_file:
    retina_dataset = json.load(retina_json_file)
    print('retina_dataset.keys()', retina_dataset.keys())
    print('retina_dataset[images]', retina_dataset['images'][0])
    image_length = len(retina_dataset['images'])
    dataset_length = len(retina_dataset['dataset'])
cur_retina_im = {}
cur_eval_im = {}
for i in range(image_length):
	cur_retina_im = retina_dataset['images'][i]
	if cur_retina_im['split'] == 'test':
		cur_eval_im['id'] = cur_retina_im['imgid']
		cur_eval_im['image_id'] = cur_retina_im['imgid']
		cur_eval_im['caption'] = cur_retina_im['sentences'][0]['raw']
json.dump(cur_eval_im, 'retina_resize_eval.json', 'w')
