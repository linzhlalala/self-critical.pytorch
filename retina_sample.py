import pandas as pd
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
freport = "/media/hdd/data/imcaption/retina_dataset_resize/label.xlsx"
reportlist = pd.read_excel(freport)
reportlist.fillna('',inplace=True)

words = []
sentences = []
prefix_path = '/media/hdd/data/imcaption/retina_dataset_resize/out'
newsize = (256, 256) 
for index, row in tqdm(reportlist.iterrows()):
    folder_name  = row['id']
    finding = row['Findings'].encode(encoding='utf-8').decode(encoding='utf-8')
    impression = row['Impression'].encode(encoding='utf-8').decode(encoding='utf-8')
    # print('folder_name', folder_name)
    mypath = os.path.join(prefix_path, row['id'])
    if os.path.isdir(mypath):
        # print('bingo')
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        if len(onlyfiles) == 0:
            print('empty folder')
        for f in listdir(mypath):
            full_f_path = os.path.join(mypath, f)
            image = Image.open(full_f_path)
            print('image size before resizing', image.size)
            reim = image.resize(newsize)
            # im.save(imgPath)
            print('image size after resizing', reim.size)
im1 = im1.resize(newsize)
print("rowid", row['id'])
print(onlyfiles)
