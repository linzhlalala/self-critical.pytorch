'''
filter some report element that not related to the image.
'''

from time import process_time_ns
import pandas as pd
import jieba
from requests.api import get
from tqdm import tqdm
import re
import csv

#r1 = '[a-zA-Z0-9’!"#$%&\'()（）*+,-./:;<=>?@，。；：?★、…【】《》？“”‘’！[\\]^_`{|}~ ]+'
symbol_trans = {ord(f):ord(t) for f,t in zip(
    u'，！？【】（）％＃＠＆：；—',
    u',!?[]()%#@&:;-')}
word_blacklist_replace = [';','。','!','?','(',')',',,,',',,']
word_blacklist_remove = ['“','”','‘','’','\"','\'','ou','OU','os','OS','od','OD','左眼','右眼',' ']
short_sent_blacklist_strip = ['.',':']
short_sent_blacklist = ['早期','晚期','中期','后期',\
    '迟缓','迅速','RCT','时间','造影诊断：','FFA','ICG','ICGA','请结合临床']
long_sent_blacklist = ['随时间延长']


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

def load_rep(freport):
    reportlist = pd.read_excel(freport)
    reportlist.fillna('',inplace=True)

    to_list = []
    for index,row in tqdm(reportlist.iterrows()):
        rid  = row['id']
        finding = row['Findings'].encode(encoding='utf-8').decode(encoding='utf-8')
        impression = row['Impression'].encode(encoding='utf-8').decode(encoding='utf-8')
        to_list.append({"id":rid,"Findings":finding,"Impression":impression})
    return to_list

def paragraph_worker(para):
    #long sents
    para = full2half(para).translate(symbol_trans)
    for word in word_blacklist_remove:
        para = para.replace(word,'')
    for word in word_blacklist_replace:
        para = para.replace(word,',')
    #break into long sents
    sents = re.split(r'\d[\.、](?!\d)',para)
    keep_long_sents = ''
    for sent in sents:
        flag_found = False
        if sent.strip() == '': 
            continue
        for word in long_sent_blacklist:
            if word in sent:
                flag_found = True
                break
        if flag_found == False:
            keep_long_sents += sent.strip()+','
    # break into short sents
    # set ,  as the only seperater
    sents = keep_long_sents.split(",")
    keep_short_sents = ''
    for sent in sents:
        flag_found = False
        for word in short_sent_blacklist_strip:
            sent = sent.strip(word)
        if len(sent)<2: 
            continue
        for word in short_sent_blacklist:
            if word in sent:
                flag_found = True
                break
        if flag_found == False:
            keep_short_sents += sent+','
    if keep_short_sents!='':
        keep_short_sents = keep_short_sents.rstrip(',')+'.'
    return keep_short_sents
                
def tokenize_translate_cn2tk(rep_list,output_name):
    ids = []
    sentFds = []
    cutFds = []
    sentImps = []
    cutImps = []
    words = []

    jieba.load_userdict('dictwords2.txt')
    print("-loading index finish-")
    #
    for row in tqdm(rep_list):
        id  = row['id']
        finding = row['Findings']
        impression = row['Impression']

        ids.append(id)

        sentence = paragraph_worker(finding)
        sentFds.append(sentence)
        cuts = jieba.lcut(sentence)
        cutFds.append(cuts)
        words.extend(cuts)
        
        sentence = paragraph_worker(impression)
        sentImps.append(sentence)
        cuts = jieba.lcut(sentence)
        cutImps.append(cuts)
        words.extend(cuts)
    #statics
    word_statics = {}
    for word in words:
        word_statics[word] = word_statics.get(word,0)+1
    keywords = word_statics.keys()
    #generate tokens
    wlen = len(keywords)
    varray = []
    offset = ord("a")
    for i in range(wlen):
        char3 = chr(i%26+offset)
        char2 = chr(i//26%26+offset)
        char1 = chr(i//676+offset)
        varray.append(char1+char2+char3)
    trans_dict = dict(zip(keywords,varray))
    print("Keywords:",wlen)
    #Trans    
    newlist = []
    for index, id in tqdm(enumerate(ids)):
        fdcut = cutFds[index]
        impcut = cutImps[index]

        enFindings,enImpression = "",""

        for word in fdcut:
            enFindings += trans_dict[word] + " "
        for word in impcut:
            enImpression += trans_dict[word] + " "
        
        if enFindings + enImpression!='':
            newlist.append({'id':id,
                'Findings': enFindings.rstrip(),
                'Impression': enImpression.rstrip(),
                'cnFindings': sentFds[index],
                'cnImps': sentImps[index]
            })
    print("Origin reports:",len(rep_list),"Final reports:",len(newlist))
    data = pd.DataFrame(newlist)
    data.to_csv(output_name+'_report.csv',index=False)      
    #save dict
    with open(output_name+'_dict.csv','w',encoding = 'utf-8',newline='') as dictfile: 
        writer = csv.writer(dictfile)
        for key, value in trans_dict.items():
            writer.writerow([key, value])


if __name__ == "__main__":
    freport = "label.xlsx"
    rep_list = load_rep(freport)
    output_name = "cn2tk_v2"
    tokenize_translate_cn2tk(rep_list,output_name)

