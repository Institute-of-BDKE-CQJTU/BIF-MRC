import json
from multiprocessing import context 

questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？', 'negative']

def question_postfix_in_question(question):
    for i in range(6):
        if questionPostfix[i] in question:
            return i
    return -1

datas = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']
type_data = [[], [], [], [], [], []]
for data in datas:
    index = question_postfix_in_question(data['question'])
    if index != -1:
        type_data[index].append(data)

def LCS(s1, s2):  
    """
        求最长公共子串，通过此算法定位多个相同答案的准确答案位置
    """ 
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  
    mmax=0   #最长匹配的长度  
    p=0  #最长匹配对应在s1中的最后一位  
    for i in range(len(s1)):  
        for j in range(len(s2)):  
            if s1[i]==s2[j]:  
                m[i+1][j+1]=m[i][j]+1  
                if m[i+1][j+1]>mmax:  
                    mmax=m[i+1][j+1]  
                    p=i+1  
    return mmax, p-mmax # 返回最长公共子串的长度和位置

def get_question_sentence(context_split, question, answer):
    index, mmax = -1, 0
    for i, span in enumerate(context_split):
        length, _ = LCS(span, question)
        if length > mmax and LCS(span, answer)[0]/len(answer) > 0.6:
            index = i
            mmax = length
    return index

newData = dict()
newData['version'] = '1.0'
newData['data'] = []         
        
for i in range(6):
    print(len(type_data[i]))
    for data in type_data[i]:
        data['context_split'] = data['context'].split('。')
        index = get_question_sentence(data['context_split'], data['question'], data['answer'])
        if index != -1:
            tempdata = dict()
            tempdata['sentence'] = data['context_split'][index]
            tempdata['context'] = data['context']
            tempdata['postfix'] = questionPostfix[i]
            tempdata['postfix_index'] = i
            tempdata['answer'] = data['answer']
            tempdata['answer_start'] = LCS(tempdata['sentence'], data['answer'])[1]
            newData['data'].append(tempdata)

count = len(newData['data'])
type_data = [[], [], [], [], [], []]
for i in range(count):
    type_data[int(newData['data'][i]['postfix_index'])].append(newData['data'][i])

max_number = 0
for i in range(6):
    if len(type_data[i]) > max_number:
        max_number = len(type_data[i])

# 平衡数据
def expand_list(lis, number):
    length = len(lis)
    multipy_number = number // length
    new_list = lis*multipy_number
    for i in range(number - length*multipy_number):
        new_list.append(lis[i])
    return new_list

for i in range(6):
    if len(type_data[i]) < max_number:
        type_data[i] = expand_list(type_data[i], max_number)

newData['data'].clear()

for i in range(max_number):
    for j in range(6):
        newData['data'].append(type_data[j][i])

with open('./data/newQuestionGeneration/bridge/question_predict_train.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(newData,ensure_ascii=False))

