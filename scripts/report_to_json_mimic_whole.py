#this edition take findings only
import pandas as pd
import json
import os
import numpy as np
import random
from tqdm import tqdm

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
    # if x == 0:
    #     return 'train'
    # elif x == 1:
    #     return 'val'
    # else:
    #     return 'test'
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


def main(params):
    # job_length = params["job_length"]
    length_threshold = params["length_threshold"]
    
    # report_csv = "cxr-study-list.csv"
    report_csv = "/media/hdd/data/imcaption/mimic/cxr-study-list.csv"
    # image_csv = "cxr-record-list.csv"
    image_csv = "/media/hdd/data/imcaption/mimic/cxr-record-list.csv"
    # metadata_csv = "mimic-cxr-2.0.0-metadata.csv"
    metadata_csv = "/media/hdd/data/imcaption/mimic/mimic-cxr-2.0.0-metadata.csv"
    txt_report_prefix = "/media/hdd/data/imcaption/mimic/mimic-cxr-reports"
    report_list = pd.read_csv(report_csv)
    image_list = pd.read_csv(image_csv)
    metadata_list = pd.read_csv(metadata_csv)
    final_list = []
    report_keys = {}
    cursor_image_list = 0    
    length_image_list = image_list.shape[0]
    token_length_list = []



    count = 0
    # print()
    train_num = 0
    test_num = 0
    val_num = 0
    for i in tqdm(range(report_list.shape[0])):
        # print('i', i)
        #a study
        path = report_list.loc[i, "path"]
        #report to string
        # txt_path = path.replace('files','reports')
        txt_path = os.path.join(txt_report_prefix, path)
        # print(txt_path)
        report_dict = txt2string(txt_path)

        #take "findings + impression" only, no finding or impression -> skip
        if "findings" not in report_dict and "impression" not in report_dict:
            # print('there is no finding or impression section', txt_path)
            continue
        else:
            text = report_dict.get("findings",'')+report_dict.get("impression",'')
        if text == "":
            print(xxxx)
        #length check       
        len_text = len(text.split(' '))
        if len_text > length_threshold:
            # continue
            pass
        
        tokens = [token for token in text.split(' ') if token != ""]
        if len(tokens) < 10:
            # print('token is shorter than 10', print(tokens))
            # continue
            pass
        token_length_list.append(len(tokens))
        study_id = report_list.loc[i, "study_id"]
        subject_id = report_list.loc[i, "subject_id"]
        #find corresponding image
        image_paths = []
        #locate all first image
        while cursor_image_list<length_image_list:
            image_sid = image_list.loc[cursor_image_list,"study_id"]
            if image_sid != study_id:
                cursor_image_list += 1
            else:                
                break
        #take all
        while cursor_image_list<length_image_list:
            image_sid = image_list.loc[cursor_image_list, "study_id"]
            if image_sid == study_id:
                image_paths.append(image_list.loc[cursor_image_list, "path"].replace('.dcm', '.jpg'))
                cursor_image_list += 1
            else:
                break
        if image_paths == []:
            continue
        for image in image_paths:
        #form json object 
            study = {}
            study['file_path'] = image
            study['sentids'] = [count]
            study['imgid'] = count
            study['sentences'] = [{'raw':text,'imgid':count,'sentid':count,'tokens':tokens}]
            study['study_id'] = str(study_id)
            study['subject_id'] = str(subject_id)
            # shuffle split
            label = get_label()
            if label == 0:
                study['split'] = 'train'
                train_num = train_num + 1
            elif label == 1:
                study['split'] = 'val'
                val_num = val_num + 1
            else:
                study['split'] = 'test'
                test_num = test_num + 1
            final_list.append(study)  
            #asset finish
            count += 1
        #if count == job_length:
        #   break

    # print("finish at ", i)
    print('total number', train_num+test_num+val_num,
     'train number', train_num,
      'test number', test_num,
       'val num', val_num)
    import collections
    counter = collections.Counter
    print(counter(token_length_list))
    # Generate data on commute times.
    size, scale = 1000, 10
    commutes = pd.Series(np.random.gamma(scale, size=size) ** 1.5)

    commutes.plot.hist(grid=True, bins=20, rwidth=0.9,
                       color='#607c8e')
    plt.title('Commute Times for 1,000 Commuters')
    plt.xlabel('Counts')
    plt.ylabel('Commute Time')
    plt.grid(axis='y', alpha=0.75)
    #print("Keys appear in report",report_keys)
    #form json
    # with open('data/dataset_mimic.json', 'w') as outfile:
    #     json.dump({'images': final_list, 'dataset': 'mimic-cxr-test'}, outfile)


if __name__ == '__main__':
    # print("This edition run for only findings length in (10,100) and PA view")
    main({"length_threshold": 100})
    print("Job Finish")