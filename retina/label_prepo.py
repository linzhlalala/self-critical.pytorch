import pandas as pd
import re
import jieba
from requests.api import get
from tqdm import tqdm
from seeker import seeker
import csv

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

def search_trans_local():
    with open('tokens.txt', 'r',encoding = 'utf-8') as tof:
        cwords = [line.strip('\n') for line in tof.readlines()]

    thedict = seeker()

    done_trans_ce = {}
    done_trans_ec = {}
    fail_words = []

    for word in tqdm(cwords):
        candies = thedict.local_seek(word)
        get_flag = False
        for index,ans in enumerate(candies):
            if ans not in done_trans_ec:
                #register and keep 1 on 1
                done_trans_ec[ans] = word
                done_trans_ce[word] = ans
                get_flag = True
                break
        if get_flag == False:
            #not available answer
            #record
            fail_words.append(word)

    print("Task total:", len(cwords))
    print("Task Local Fail:",len(fail_words)) 

    with open('temp_dict.csv','w', encoding = 'utf-8',newline='') as dictfile: 
        writer = csv.writer(dictfile)
        for key, value in done_trans_ce.items():
            writer.writerow([key, value])
    
    with open('failwords.txt', 'w', encoding = 'utf-8') as ffile:
        for word in fail_words:
            ffile.write(word+ '\n')
    

def search_trans_iciba():
    done_trans_ce = {}
    
    with open('temp_dict.csv','r',encoding = 'utf-8') as dictfile: 
        reader=csv.reader(dictfile,delimiter=',')
        for row in reader:
            done_trans_ce[row[0]]=row[1]
            #done_trans_ec[row[1]]=row[0]

    with open('failwords.txt', 'r',encoding = 'utf-8') as tof:
        cwords = [line.strip('\n') for line in tof.readlines()]

    thedict = seeker()
    fail_words = []

    for word in tqdm(cwords):
        ans = thedict.iciba_seek_api(word)

        if ans != "": #and ans not in done_trans_ec:
            #register and keep 1 on 1
            #done_trans_ec[ans] = word
            done_trans_ce[word] = ans
        else:
            #not available answer
            #record
            fail_words.append(word)

    print("Task total:", len(cwords))
    print("Task iciba Fail:",len(fail_words)) 

    with open('failwords.txt', 'w',encoding = 'utf-8') as ffile:
        for word in fail_words:
            ffile.write(word+ '\n')
    
    with open('temp_dict.csv','w',encoding = 'utf-8',newline='') as dictfile: 
        writer = csv.writer(dictfile)
        for key, value in done_trans_ce.items():
            writer.writerow([key, value])

def manual_fixing_dict():
    done_trans_ce = {}
    
    with open('temp_dict.csv','r',encoding = 'utf-8') as dictfile: 
        reader=csv.reader(dictfile,delimiter=',')
        for row in reader:
            done_trans_ce[row[0]]=row[1].lower()
            if row[1] == '':
                print(row[0])
            #done_trans_ec[row[1]]=row[0]
    #manual fix
    done_trans_ce['周强'] = 'surrounded by strong'
    done_trans_ce['尅'] = '' #打错的字
    done_trans_ce['良好'] = 'well'
    done_trans_ce['簇状强'] = 'cluster-like strong'
    done_trans_ce['疤痕'] = 'scar'
    done_trans_ce['未能'] = 'cannot'
    done_trans_ce['盘周'] = 'disk around'
    done_trans_ce['稍直'] = 'slightly straight'
    done_trans_ce['瘤样'] = 'tumor-like'
    done_trans_ce['弧余'] = 'arc left'
    done_trans_ce['怒张'] = 'serious open'
    done_trans_ce['较弱'] = 'weaker'
    done_trans_ce['状拟'] = 'shape like'
    done_trans_ce['局灶弱'] = 'partly lesion weak'
    done_trans_ce['弧低'] = 'arc lower'
    done_trans_ce['颞下经'] = 'temporal inferior through'
    done_trans_ce['呈瘤样'] = 'be tumor-like'
    done_trans_ce['较直'] = 'more straight'
    done_trans_ce['较细仅'] = 'thinner only'
    done_trans_ce['细细'] = 'thinner'
    done_trans_ce['之透见'] = 'look through'
    done_trans_ce['处小簇'] = 'small cluster'
    done_trans_ce['晕环'] = 'halo ring'
    done_trans_ce['弧鼻'] = 'arc nose' 
    done_trans_ce['簇状轻'] = 'cluster-like slight'
    done_trans_ce['灶弱'] = 'lesion weak'
    done_trans_ce['鞥'] = '' #打错的字
    done_trans_ce['数小簇'] = 'many small cluster'
    done_trans_ce['两簇'] = 'two clusters of'
    done_trans_ce['绕低'] = 'surrounded by low'
    done_trans_ce['忋'] = '' #改 打错的
    done_trans_ce['周部散'] = 'surrounded by scattered'
    done_trans_ce['周小丛'] = 'surrounded by small clusters'
    #错译, 全靠偶然看见...
    done_trans_ce['中大'] = 'center large' #CUHK
    done_trans_ce['余'] = 'left' #I
    done_trans_ce['性异常'] = 'type abnormality' #sexual abnormality
    done_trans_ce['性的'] = 'type' #sexual
    done_trans_ce['性富'] = 'type rich'#sexual wealth
    done_trans_ce['周五'] = 'surrounded by five' #friday
    done_trans_ce['周部团'] = 'surrounded by group' #ministry of foreign affairs
    done_trans_ce['部对'] = 'section pair' #,ministry of foreign affairs
    done_trans_ce['部约'] = 'section about' #,ministry of foreign affairs
    done_trans_ce['部数'] = 'section several' #,number of departments
    done_trans_ce['部团'] = 'section group'#,department
    done_trans_ce['部连'] = 'section junction'#,department
    done_trans_ce['部早'] = 'section early'#,department
    done_trans_ce['部及'] = 'section and'#department
    done_trans_ce['部多'] = 'section many'#number of departments
    done_trans_ce['部见点'] = 'section shows point'#department
    done_trans_ce['部点'] = 'section points'#ministry points
    done_trans_ce['部见局'] = 'section shows some'#department bureau
    done_trans_ce['周部视'] = 'section shows'#weekly department
    done_trans_ce['部多局'] = 'section shows many'#,multi-department
    done_trans_ce['部见团'] = 'section shows group'#department members
    done_trans_ce['部及视'] = 'section and shows'#department and vision
    done_trans_ce['部视'] = 'section shows'#department
    done_trans_ce['部处'] = 'section shows'#department
    done_trans_ce['部以'] = 'section' #ministry
    done_trans_ce['球样'] = 'ball-like'#sample
    done_trans_ce['周部于'] = 'surrounded by'#ministry zhou
    done_trans_ce['性病变'] = 'type lesion'#std
    done_trans_ce['Ω'] = 'Ω' #欧盟
    done_trans_ce['英光区'] = 'fluorescence region'#yingguang district
    done_trans_ce['眼见'] = 'finds'#see you
    done_trans_ce['区约'] = 'area about'#district councils
    done_trans_ce['远中'] = 'far away from center'#fa
    done_trans_ce['湖样'] = 'lake-like'#湖样,lake sample
    done_trans_ce['不随'] = 'not with' #"no, no"
    done_trans_ce['左眼'] = 'left eye' # ''

    with open('temp_dict.csv','w',encoding = 'utf-8',newline='') as dictfile: 
        writer = csv.writer(dictfile)
        for key, value in done_trans_ce.items():
            writer.writerow([key, value])

def rough_translate():
    #report
    freport = "label.xlsx"
    reportlist = pd.read_excel(freport)
    reportlist.fillna('',inplace=True)
    #dict_index
    jieba.load_userdict('dictwords.txt')
    print("-loading index finish-")
    #dict
    dict_ce = {}
    with open('temp_dict.csv','r',encoding = 'utf-8') as dictfile: 
        reader=csv.reader(dictfile,delimiter=',')
        for row in reader:
            dict_ce[row[0]]=row[1]

    print("-loading dict finish-")

    symbol_trans = {ord(f):ord(t) for f,t in zip(
        u'，。！？【】（）％＃＠＆“”、‘’：',
        u',.!?[]()%#@&\"\",\'\':')}

    def single_line_trans(sentence):
        sentence = full2half(sentence).translate(symbol_trans)
        cuts = jieba.lcut(sentence)
        enSentence = ''        
        lanFlag = True #True for cn , False for non cn
        #Flags to identify the use of space
        for word in cuts:
            if word in dict_ce:
                enWord = dict_ce[word]
                #cncn -> en en ,#encn -> en en
                enSentence += ' '+enWord
                lanFlag = True
            else:
                #cnen-> en en, #enen->enen
                if lanFlag and word not in u',.!?[]()%#@&\"\",\'\':':
                    enSentence += ' '+word
                else:
                    enSentence += word
                lanFlag = False
        return enSentence.lstrip()
    
    newlist = []
    for index, row in tqdm(reportlist.iterrows()):
        id  = row['id']
        finding = row['Findings'].encode(encoding='utf-8').decode(encoding='utf-8')
        impression = row['Impression'].encode(encoding='utf-8').decode(encoding='utf-8')
        
        enFindings = single_line_trans(finding)
        enImpression = single_line_trans(impression)

        newlist.append({'id':id,
            'Findings': enFindings,
            'Impression': enImpression
        })
    data = pd.DataFrame(newlist)
    data.to_csv('rtrans_reports.csv',index=False)

def dictword_reduce():
    with open('backup_en\\tokens.txt', 'r',encoding = 'utf-8') as tof:
        tokens = [line.strip('\n') for line in tof.readlines()]
    with open('backup_en\\dictwords.txt', 'r',encoding = 'utf-8') as tof:
        dictwords = [line.strip('\n') for line in tof.readlines()]

    print("Reduce Task, tokens:{}, dictwords:{}".format(len(tokens),len(dictwords)))

    retword = [word for word in dictwords if word in tokens]
    print("Reduce Task, retword:{}".format(len(retword)))

    with open('retwords.txt', 'w',encoding = 'utf-8') as ffile:
        for word in retword:
            ffile.write(word+ '\n')

    diffword =  [word for word in tokens if word not in retword]
    print("Reduce Task, diffword:{}".format(len(diffword)))
    with open('diffwords.txt', 'w',encoding = 'utf-8') as ffile:
        for word in diffword:
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

def tokenize_translate():
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
        
if __name__ == "__main__":
    #clean()
    #print("clean finish")
    #dictword_reduce()
    #manual_fixing_tokens1()
    #cut()
    #print("cut finish")
    #search_trans_local()
    #search_trans_iciba()
    #print("search finish")
    #search_trans_second()
    #manual_fixing_dict()
    #rough_translate()
    tokenize_translate()
