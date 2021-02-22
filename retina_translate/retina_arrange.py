import pandas as pd
from tqdm import tqdm
import json
import os
import random
import skimage.io

def get_label():
    sampleList = [0, 0, 0, 0, 0, 0, 0, 1, 1, 2] #: 0 train 1 validate 2 test
    x = random.choice(sampleList)

    return x

def main():
    #cut it
    freport = "retina_translate/cn2tk_v2_report.csv"
    imgfolder = "../retina_wp/select0.625"
    reportlist = pd.read_csv(freport)
    reportlist.fillna('',inplace=True)

    new_list = []
    count_id = 0
    train_num = 0
    val_num = 0
    test_num = 0
    error_file_count = 0
    for index, row in tqdm(reportlist.iterrows()):
        id  = row['id']
        finding = row['Findings']
        impression = row['Impression']

        report = "finding: " + finding + " impression: " + impression
        tokens = report.split(" ")

        study_dir = os.path.join(imgfolder,id)

        if not os.path.exists(study_dir):
            #doesn't have this study data
            continue

        images= os.listdir(study_dir) 
        for image in images:
            image_file = os.path.join(study_dir,image)
            #try:
                #I = skimage.io.imread(image_file)
            #except:
                #error_file_count += 1
                #continue
            record = {}
            record['filepath'] = id
            record['filename'] = image
            record['sentids'] = [count_id]
            record['imgid'] = count_id
            record['sentences'] = [{'raw':report,'imgid':count_id,'sentid':count_id,'tokens':tokens}]
            label = get_label()
            if label == 0:
                record['split'] = 'train'
                train_num = train_num + 1
            elif label == 1:
                record['split'] = 'val'
                val_num = val_num + 1
            else:
                record['split'] = 'test'
                test_num = test_num + 1
            count_id += 1
            new_list.append(record)
            
    print("convert totally:{} reports, {} images".format(len(reportlist),count_id))
    print("train:{}, val:{}, test:{}".format(train_num,val_num,test_num))
    #print("error file: {}".format(error_file_count))
    with open("dataset_retina5.json","w") as output:
        json.dump({"images":new_list,'dataset':"retina5"},output)

if __name__ == "__main__":
    main()
