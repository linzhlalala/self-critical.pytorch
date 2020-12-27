import pandas as pd
import re
import jieba
from requests.api import get
from tqdm import tqdm
#from seeker import seeker
import csv
import os
from os import listdir
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


r1 = '[a-zA-Z0-9’!"#$%&\'()（）*+,-./:;<=>?@，。；：?★、…【】《》？“”‘’！[\\]^_`{|}~ ]+'

#全角转成半角
def full2half(s):
    n = ''
    for char in s:
        num = ord(char)
        if num == 0x3000:        #将全角空格转成半角空格
            num = 32
        elif 0xFF01 <=num <= 0xFF5E:       #将其余全角字符转成半角字符
            num -= 0xFEE0
        num = chr(num)
        n += num
    return n

def clean():
    #freport = "sample.csv"
    freport = "label.xlsx"
    reportlist = pd.read_excel(freport)
    reportlist.fillna('',inplace=True)

    ftxt = open('labels.txt', 'w',encoding = 'utf-8')

    for index, row in tqdm(reportlist.iterrows()):
        #id  = row['id']
        finding = row['Findings'].encode(encoding='utf-8').decode(encoding='utf-8')
        impression = row['Impression'].encode(encoding='utf-8').decode(encoding='utf-8')

        finding = full2half(finding)
        impression = full2half(impression)

        finding = (re.sub(r1, '', finding))
        impression = (re.sub(r1, '', impression))

        if finding !="":
            ftxt.write(finding+ '\n')
        if impression !="":
            ftxt.write(impression+ '\n')
    ftxt.close()

def cut():
    freport = "label.xlsx"
    reportlist = pd.read_excel(freport)
    reportlist.fillna('',inplace=True)
    
    jieba.load_userdict('dictwords.txt')
    jieba.add_word('1.')
    jieba.add_word('2.')
    jieba.add_word('3.')
    jieba.add_word('4.')
    jieba.add_word('5.')

    
    words = []
    sentences = []
    for index, row in tqdm(reportlist.iterrows()):
        #id  = row['id']
        finding = row['Findings'].encode(encoding='utf-8').decode(encoding='utf-8')
        impression = row['Impression'].encode(encoding='utf-8').decode(encoding='utf-8')

        cuts = jieba.lcut(finding)
        words.extend(cuts)
        sentences.append('/'.join(cuts))

        cuts = jieba.lcut(impression)
        words.extend(cuts)
        sentences.append('/'.join(cuts))

    word_statics = {}

    for word in words:
        word_statics[word] = word_statics.get(word,0)+1

    #print(word_statics)
    tokens = word_statics.keys()
    with open('tokens.txt', 'w',encoding='utf-8') as f:
        for token in tokens:
            f.write(token+'\n')
    with open('cut_test.txt', 'w',encoding='utf-8') as f:
        for stc in sentences:
            f.write(stc+'\n')
    
    with open('dictwords.txt', 'r',encoding = 'utf-8') as tof:
        cwords = [line.strip('\n') for line in tof.readlines()]
    
    diffwords = [word for word in tokens if word not in cwords]

    with open('diffwords.txt', 'w',encoding = 'utf-8') as ffile:
        for word in diffwords:
            ffile.write(word+ '\n')


def manual_fixing_tokens1():
    with open('retwords.txt', 'r',encoding = 'utf-8') as tof:
        tokens = [line.strip('\n') for line in tof.readlines()]

    tokens.append("小丛")
    tokens.append("黄斑区")
    tokens.append("后极部")
    tokens.append("囊样")
    tokens.append("中周部")
    tokens.append("灌注区")
    tokens.append("花瓣样")
    tokens.append("湖状")
    tokens.append("萎缩灶")
    tokens.append("散在")
    tokens.append("中周部")
    tokens.append("中周")
    tokens.append("中晚期")
    
    with open('dictwords.txt', 'w',encoding = 'utf-8') as ffile:
        for word in tokens:
            ffile.write(word+ '\n')

def manual_fixing_tokens2():
    with open('retwords.txt', 'r',encoding = 'utf-8') as tof:
        tokens = [line.strip('\n') for line in tof.readlines()]

    tokens.append("屈光间质")
    tokens.append("不清")
    tokens.append("弱荧光")
    tokens.append("无灌注区")
    tokens.append("为主")
    tokens.append("为甚")
    tokens.append("为著")
    tokens.append("多发")
    
    with open('dictwords.txt', 'w',encoding = 'utf-8') as ffile:
        for word in tokens:
            ffile.write(word+ '\n')

def tokenize_translate_cn2tk():
    #cut it
    freport = "label.xlsx"
    reportlist = pd.read_excel(freport)
    reportlist.fillna('',inplace=True)
    
    jieba.load_userdict('dictwords.txt')
    jieba.add_word('1.')
    jieba.add_word('2.')
    jieba.add_word('3.')
    jieba.add_word('4.')
    jieba.add_word('5.')
    print("-loading index finish-")

    symbol_trans = {ord(f):ord(t) for f,t in zip(
        u'，。！？【】（）％＃＠＆“”、‘’：',
        u',.!?[]()%#@&\"\",\'\':')}

    ids = []
    cutFds = []
    cutImps = []
    words = []
    for index, row in tqdm(reportlist.iterrows()):
        id  = row['id']
        finding = row['Findings'].encode(encoding='utf-8').decode(encoding='utf-8')
        impression = row['Impression'].encode(encoding='utf-8').decode(encoding='utf-8')

        ids.append(id)

        sentence = full2half(finding).translate(symbol_trans)
        cuts = jieba.lcut(sentence)
        cutFds.append(cuts)
        words.extend(cuts)
        
        sentence = full2half(impression).translate(symbol_trans)
        cuts = jieba.lcut(sentence)
        cutImps.append(cuts)
        words.extend(cuts)
    #statics
    word_statics = {}

    for word in words:
        word_statics[word] = word_statics.get(word,0)+1
    keywords = word_statics.keys()
    #print(word_statics)
    #form dict
    #dictword 1999, diffworks 1349 
    wlen = len(keywords)
    varray = []
    offset = ord("a")
    for i in range(wlen):
        char3 = chr(i%26+offset)
        char2 = chr(i//26%26+offset)
        char1 = chr(i//676+offset)
        varray.append(char1+char2+char3)
    trans_dict = dict(zip(keywords,varray))
    #trans
    
    newlist = []
    for index, id in tqdm(enumerate(ids)):
        fdcut = cutFds[index]
        impcut = cutImps[index]

        enFindings,enImpression = "",""

        for word in fdcut:
            enFindings += trans_dict[word] + " "
        for word in impcut:
            enImpression += trans_dict[word] + " "

        newlist.append({'id':id,
            'Findings': enFindings.rstrip(),
            'Impression': enImpression.rstrip(),
        })
    data = pd.DataFrame(newlist)
    data.to_csv('token_trans_reports.csv',index=False)      
    #save dict
    with open('token_trans_dict.csv','w',encoding = 'utf-8',newline='') as dictfile: 
        writer = csv.writer(dictfile)
        for key, value in trans_dict.items():
            writer.writerow([key, value])
        
def tokenize_translate_tk2cn():
    freport = "data/token_trans_reports.csv"
    reportlist = pd.read_csv(freport)
    reportlist.fillna('',inplace=True)
    print(reportlist)
    dict_tk2cn = {}
    with open('data/token_trans_dict.csv','r',encoding = 'utf-8') as dictfile: 
        reader=csv.reader(dictfile,delimiter=',')
        for row in reader:
            dict_tk2cn[row[1]]=row[0]

    def tk2cn(tks):
        if tks =="": return ""
        ltk = tks.split(" ")
        ans = ""
        for tk in ltk:
            ans += dict_tk2cn[tk]
        return ans

    newlist = []
    for index, row in tqdm(reportlist.iterrows()):
        # print(index, row)
        id  = row['id']
        finding = row['Findings']
        impression = row['Impression']

        cnFds = tk2cn(finding)
        cnImp = tk2cn(impression)

        newlist.append({'id':id,
            'Findings': cnFds,
            'Impression': cnImp,
        })
        if index == 10:
            break
    print(newlist)

def im_address():
    freport = "data/token_trans_reports.csv"
    reportlist = pd.read_csv(freport)
    reportlist.fillna('', inplace=True)
    prefix_path = '/media/hdd/data/imcaption/retina_dataset_resize/resize'
    finding_counter = []

    dict_tk2cn = {}
    with open('data/token_trans_dict.csv','r',encoding = 'utf-8') as dictfile: 
        reader=csv.reader(dictfile,delimiter=',')
        for row in reader:
            dict_tk2cn[row[1]]=row[0]
    def tk2cn(tks):
        if tks =="": return ""
        ltk = tks.split(" ")
        ans = ""
        for tk in ltk:
            ans += dict_tk2cn[tk]
        return ans
    # load the coding example 
    for index, row in tqdm(reportlist.iterrows()):
        i = index
        folderpath = os.path.join(prefix_path, reportlist.iloc[index]['id'])
        if os.path.exists(folderpath):
            for f in listdir(folderpath):
                # print(f)
                ab_f_path = os.path.join(folderpath, f)
                # print(reportlist.iloc[index]['Findings'], ab_f_path)
                finding = reportlist.iloc[index]['Findings']
                impression = reportlist.iloc[index]['Impression']
                finding_token = finding.split(" ")
                finding_length = len(finding_token)
                # print('finding_length', finding_length)
                if finding_length < 5:
                    # for token_f in finding:
                    #     print('token_f', token_f)
                    cn_finding = tk2cn(finding)
                    print('cn_finding', cn_finding, 'finding', finding)
                finding_counter.append(finding_length)
        # if index == 4000:
        #     break
    cn_finding = tk2cn(finding)
    cn_impression = tk2cn(impression)
    # print('finding convcerted back to Chinese', cn_finding)
    # print(index, row['id'])
    # print('total number of images', len(finding_counter))
    finding_distribution = Counter(finding_counter)
    print(Counter(finding_counter))
    # plt.hist(finding_counter, bins=20)
    # plt.show()
    # print(dict_tk2cn)
    # print('report length distribution', print(Counter(finding_counter)))
if __name__ == "__main__":
    # tokenize_translate_tk2cn()
    im_address()