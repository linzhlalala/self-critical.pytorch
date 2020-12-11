#this edition take findings only
import pandas as pd
import json
import os
import numpy as np
import random
from tqdm import tqdm
import re

def txt2string(fpath):
    f = open(fpath,'r')
    lines = f.readlines()
    #process here if required
    finaldict = {}

    last_key = ""
    for line in lines:
        line = line.strip().lower()
        if line !='':
            if ':' in line:
                key, value = line.split(':', 1)
                if key != '' and not key[-1].isnumeric():
                    finaldict[key] = value.strip()
                    last_key = key
                elif last_key != "":
                    finaldict[last_key] += ' ' + line
            elif last_key != "":
                finaldict[last_key] += ' ' + line
    
    #print(finaldict)
    return finaldict


def get_label():
    sampleList = [0, 0, 0, 0, 0, 0, 0, 1, 1, 2] #: 0 train 1 validate 2 test
    x = random.choice(sampleList)
    return x

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

file_name = '/media/hdd/donghao/imcaption/R2Gen/data/mimic_cxr/annotation.json'
# file_name = '/media/hdd/donghao/imcaption/R2Gen/data/iu_xray/annotation.json'
with open(file_name) as json_file:
    data = json.load(json_file)
    keys = data.keys()

final_list = []
count = 0
train_report_num = 0
val_report_num = 0
test_report_num = 0
for each_item in data['train']:
    train_report_num = train_report_num + 1 
    text = each_item['report'] 
    text = clean_report_mimic_cxr(text)
    tokens = [token for token in text.split(' ') if token != ""]
    # if len(each_item['image_path']) > 1:
    #     print('bingo')
    for impath in each_item['image_path']:
        count = count + 1
        study = {}
        study['file_path'] = impath
        study['sentids'] = [count]
        study['imgid'] = count
        study['sentences'] = [{'raw':text,'imgid':count,'sentid':count,'tokens':tokens}]
        study['study_id'] = each_item['study_id']
        study['subject_id'] = each_item['subject_id']
        study['split'] = each_item['split']
        final_list.append(study)

for each_item in data['test']:
    test_report_num = test_report_num + 1
    text = each_item['report'] 
    text = clean_report_mimic_cxr(text)
    tokens = [token for token in text.split(' ') if token != ""]
    count = count + 1
    # if len(each_item['image_path']) > 1:
    #     print('bingo')
    for impath in each_item['image_path']:
        count = count + 1
        study = {}
        study['file_path'] = impath
        study['sentids'] = [count]
        study['imgid'] = count
        study['sentences'] = [{'raw':text,'imgid':count,'sentid':count,'tokens':tokens}]
        study['study_id'] = each_item['study_id']
        study['subject_id'] = each_item['subject_id']
        study['split'] = each_item['split']
        final_list.append(study)

for each_item in data['val']:
    val_report_num = val_report_num + 1
    text = each_item['report'] 
    text = clean_report_mimic_cxr(text)
    tokens = [token for token in text.split(' ') if token != ""]
    # if len(each_item['image_path']) > 1:
    #     print('bingo')

    for impath in each_item['image_path']:
        count = count + 1
        study = {}
        study['file_path'] = impath
        study['sentids'] = [count]
        study['imgid'] = count
        study['sentences'] = [{'raw':text,'imgid':count,'sentid':count,'tokens':tokens}]
        study['study_id'] = each_item['study_id']
        study['subject_id'] = each_item['subject_id']
        study['split'] = each_item['split']
        final_list.append(study)

print('the number of studies', len(final_list))
print('train_image_num', train_report_num, 'val_image_num', val_report_num, 'test_image_num', test_report_num)
# print(each_item['study_id'], each_item['subject_id'])
# print(data['val'][30])
# print(data['val'][31])
# print(data['val'][32])
# print(data['val'][33])
# print(data['test'][33])
# print(data['test'][34])
# print(data['test'][35])
# print(data['test'][36])
print(keys)



