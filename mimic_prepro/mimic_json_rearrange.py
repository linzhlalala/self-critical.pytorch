import json
import os

def main(anno_file):
    js_file = json.load(open(anno_file,'r'))
    report_list = list(js_file["train"])
    report_list.extend(list(js_file["val"]))
    report_list.extend(list(js_file["test"]))
    new_list = []

    count_id = 0
    for report in report_list:
        folder = report["id"]
        text = report["report"]
        tokens = [word for word in text.replace("."," .").replace(","," ,").replace('\n','').split(" ")]
        textlen = len(tokens)
        split = report["split"]
        images = report["image_path"]
        for image in images:
            image_name = image.split("/")
            #build new record
            study = {}
            study['filepath'] = '/'.join(image_name[0:3])
            study['filename'] = image_name[3]
            study['sentids'] = [count_id]
            study['imgid'] = count_id
            study['sentences'] = [{'raw':text,'imgid':count_id,'sentid':count_id,'tokens':tokens}]
            study['split'] = split
            count_id += 1
            new_list.append(study)
    print("convert totally:{} reports, {} images".format(len(report_list),count_id))

    with open("..//mimic_cxr_256//dataset_m256.json","w") as output:
        json.dump({"images":new_list,'dataset':"iu_xray"},output)

if __name__ == '__main__':
    iu_anno_path = os.path.join("..","mimic_cxr_256","annotation.json")
    anno_file = main(iu_anno_path)



