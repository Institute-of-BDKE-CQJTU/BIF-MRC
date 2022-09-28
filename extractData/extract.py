import re
import os
import pandas as pd

def read(path):
    vv= []
    for i in os.listdir(path):
        fi_d = os.path.join(path, i)
        if os.path.isdir(fi_d):
        	# 调用递归
            read(fi_d)
        else:
            if os.path.splitext(i)[1]=='.log':
                vv.append(os.path.join(fi_d))
    return vv

# 抽取ours的EM和F1值
def extractFewShotEMAndF1(domain):
    logname = ['log1', 'log2', 'log3', 'log4', 'log5', 'logfull']
    seed = [13, 43, 83, 181, 271, 347, 433, 659, 727, 859]
    f_out = open('./extractData/tempData.csv', 'w', encoding='UTF-8')
    f_out.write('domain,epochs,iter,样本数,F1,EM'+'\n') 
    for part in ['ours']:
        for j in range(1, 4):
            for k in range(10):
                files = read('./log/'+domain+'/'+part+'/'+str(j*10)+'/'+str(seed[k])+'/')
                for file in files:
                    with open(file, 'r', encoding='UTF-8') as f:
                        text = f.readlines()[-1]
                        scores = re.search(r'F1:\d+(\.\d+),EM:\d+(\.\d+)', text).group()
                        number = re.search(r'\d+', file[-9:]).group()
                        if int(number) < 16:
                            f_out.write('{},10,{},{},{},{}'.format(domain+part, j, 2844, scores[3:9], scores[13:]))
                        else:
                            f_out.write('{},10,{},{},{},{}'.format(domain+part, j, number, scores[3:9], scores[13:]))
                        f_out.write('\n')
    for part in ['finetune', 'bart_finetune', 'splinter', 'review', 'macbert_finetune', 'roberta_finetune', 'question_type_random', 'question_type_random_new', 'random_answer', 'all_random', 'no_postfix']:
        for j in range(2, 3):
            for k in range(10):
                files = read('./log/'+domain+'/'+part+'/'+str(j)+'/'+str(seed[k])+'/')
                for file in files:
                    with open(file, 'r', encoding='UTF-8') as f:
                        text = f.readlines()[-1]
                        scores = re.search(r'F1:\d+(\.\d+),EM:\d+(\.\d+)', text).group()
                        number = re.search(r'\d+', file[-9:]).group()
                        if int(number) < 16:
                            f_out.write('{},10,{},{},{},{}'.format(domain+part, j, 2844, scores[3:9], scores[13:]))
                        else:
                            f_out.write('{},10,{},{},{},{}'.format(domain+part, j, number, scores[3:9], scores[13:]))
                        f_out.write('\n')
    for part in ['ours_3', 'ours_4', 'ours_5', 'ours_6', 'ours_4_bart', 'ours_5_bart', 'ours_6_bart', 'ours_4_macbert', 'ours_5_macbert', 'ours_6_macbert', 'ours_4_roberta', 'ours_5_roberta', 'ours_6_roberta']:
        for j in [1, 2]:
            for k in range(10):
                files = read('./log/'+domain+'/'+part+'/'+str(j)+'/'+str(seed[k])+'/')
                for file in files:
                    with open(file, 'r', encoding='UTF-8') as f:
                        text = f.readlines()[-1]
                        scores = re.search(r'F1:\d+(\.\d+),EM:\d+(\.\d+)', text).group()
                        number = re.search(r'\d+', file[-9:]).group()
                        if int(number) < 16:
                            f_out.write('{},10,{},{},{},{}'.format(domain+part, j, 2844, scores[3:9], scores[13:]))
                        else:
                            f_out.write('{},10,{},{},{},{}'.format(domain+part, j, number, scores[3:9], scores[13:]))
                        f_out.write('\n')
    for part in ['ours_4_macbert', 'ours_5_macbert', 'ours_6_macbert', 'ours_4_roberta', 'ours_5_roberta', 'ours_6_roberta']:
        for j in [3]:
            for k in range(10):
                files = read('./log/'+domain+'/'+part+'/'+str(j)+'/'+str(seed[k])+'/')
                for file in files:
                    with open(file, 'r', encoding='UTF-8') as f:
                        text = f.readlines()[-1]
                        scores = re.search(r'F1:\d+(\.\d+),EM:\d+(\.\d+)', text).group()
                        number = re.search(r'\d+', file[-9:]).group()
                        if int(number) < 16:
                            f_out.write('{},10,{},{},{},{}'.format(domain+part, j, 2844, scores[3:9], scores[13:]))
                        else:
                            f_out.write('{},10,{},{},{},{}'.format(domain+part, j, number, scores[3:9], scores[13:]))
                        f_out.write('\n')
    for part in ['0.2', '0.4', '0.6', '0.8']:
        for j in range(1, 2):
            for k in range(10):
                files = read('./log/'+domain+'/part_6_macbert/'+part+'/'+str(j)+'/'+str(seed[k])+'/')
                for file in files:
                    with open(file, 'r', encoding='UTF-8') as f:
                        text = f.readlines()[-1]
                        scores = re.search(r'F1:\d+(\.\d+),EM:\d+(\.\d+)', text).group()
                        number = re.search(r'\d+', file[-9:]).group()
                        if int(number) < 16:
                            f_out.write('{},10,{},{},{},{}'.format(domain+'6_macbert_'+part, j*2, 2844, scores[3:9], scores[13:]))
                        else:
                            f_out.write('{},10,{},{},{},{}'.format(domain+'6_macbert_'+part, j*2, number, scores[3:9], scores[13:]))
                        f_out.write('\n')
    f_out.close()                    


def groupData():
    data = pd.read_csv('./extractData/tempData.csv')
    newdata = data.groupby(['domain', 'epochs', 'iter', '样本数']).mean()
    newdata.to_csv('./extractData/finalData.csv')

if __name__ == "__main__":
    extractFewShotEMAndF1('bridge')
    groupData()