#this edition take findings only
import pandas as pd
import json
import os
import numpy as np


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


def main(params):
    job_length = params["job_length"]
    length_threshold = params["length_threshold"]
    
    report_csv = "cxr-study-list.csv"
    image_csv = "cxr-record-list.csv"
    metadata_csv = "mimic-cxr-2.0.0-metadata.csv"

    report_list = pd.read_csv(report_csv)
    image_list = pd.read_csv(image_csv)
    metadata_list = pd.read_csv(metadata_csv)

    report_keys = {}
    cursor_image_list = 0

    val_split = int(0.2*job_length)
    test_split = int(0.1*job_length)
    
    final_list = []

    split = np.zeros(job_length)
    split[:val_split] = 1
    split[-test_split:] = 2
    np.random.shuffle(split)

    count = 0
    for i in range(report_list.shape[0]):
        #a study
        path = report_list.loc[i,"path"]
        #report to string
        txt_path = path.replace('files','reports')
        report_dict = txt2string(txt_path)

        #take findings only, no finding = skip
        if "findings" in report_dict:
            text = report_dict["findings"]
        else:
            continue
        #length check       
        len_text = len(text.split(' '))
        if len_text > length_threshold:
            continue
        
        tokens = [token for token in text.split(' ') if token != ""]
        if len(tokens) < 10:
            continue

        study_id = report_list.loc[i,"study_id"]
        subject_id = report_list.loc[i,"subject_id"]
        #find corresponding image
        image_paths = []
        #locate
        while True:
            image_sid = image_list.loc[cursor_image_list,"study_id"]
            if image_sid != study_id:
                cursor_image_list += 1
            else:                
                break
        #take
        while True:
            image_sid = image_list.loc[cursor_image_list, "study_id"]
            view_position = metadata_list.loc[cursor_image_list, "ViewPosition"]
            if image_sid == study_id:
                if view_position == "PA":
                    image_paths.append(image_list.loc[cursor_image_list, "path"].replace('.dcm', '.jpg'))
                cursor_image_list += 1
            else:
                break
        if image_paths == []:
            continue
        elif len(image_paths) != 1:
            pass
            #print("warning: report {} has more than 1 PA".format(i))
            #print(image_paths[0])
        #form json object 
        study = {}
        study['file_path'] = image_paths[-1]#usually the last one is better position image when there are many
        study['sentids'] = [count]
        study['imgid'] = count
        study['sentences'] = [{'raw':text,'imgid':count,'sentid':count,'tokens':tokens}]
        study['study_id'] = str(study_id)
        study['subject_id'] = str(subject_id)
        # shuffle split
        if split[count] == 0:
            study['split'] = 'train'
        elif split[count] == 1:
            study['split'] = 'val'
        else:
            study['split'] = 'test'
        final_list.append(study)  
        #asset finish
        count += 1
        if count == job_length:
            break

    print("finish at ",i)
    #print("Keys appear in report",report_keys)
    #form json
    with open('findings.json', 'w') as outfile:
        json.dump({'images':final_list,'dataset':'mimic-cxr-test'}, outfile)


if __name__ == '__main__':
    print("This edition run for only findings length in (10,50) and PA view")
    main({"job_length": 2000, "length_threshold": 50})
    print("Job Finish")
