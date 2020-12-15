import pandas as pd
from tqdm import tqdm
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import multiprocessing
freport = "/media/hdd/data/imcaption/retina_dataset_resize/label.xlsx"
reportlist = pd.read_excel(freport)
reportlist.fillna('',inplace=True)
# words = []
# sentences = []
# print(len(reportlist)) 
# print(reportlist.iloc[3]['id'])

def resize_folder(row_number):
    prefix_path = '/media/hdd/data/imcaption/retina_dataset_resize/out'
    output_path = '/media/hdd/data/imcaption/retina_dataset_resize/resize'
    newsize = (256, 256)
    mypath = os.path.join(prefix_path, reportlist.iloc[row_number]['id'])
    savepath = os.path.join(output_path, reportlist.iloc[row_number]['id'])
    if os.path.isdir(mypath):
        try:
            os.stat(savepath)
        except:
            os.mkdir(savepath) 
        # print('bingo')
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        if len(onlyfiles) == 0:
            print('empty folder')
        for f in listdir(mypath):
            full_f_path = os.path.join(mypath, f)
            image = Image.open(full_f_path)
            # print('image size before resizing', image.size)
            reim = image.resize(newsize)
            # im.save(imgPath)
            # print('image size after resizing', reim.size)
            imsavepath = os.path.join(savepath, f)
            # print('the current image being saved is', imsavepath)
            reim.save(imsavepath)
        print('the number of images in this folder', len(onlyfiles)) 
    return None
jobs = []
print('the length of report list', len(reportlist))
# for i in range(0, 100, 1):
# for i in range(100, 1000, 1):
# for i in range(1000, 1500, 1):
# for i in range(3000, 4000, 1):
# for i in range(9000, 10000, 1):
for i in range(10000, 10302, 1):
    p = multiprocessing.Process(target=resize_folder, args=(i,))
    jobs.append(p)
    p.start()

