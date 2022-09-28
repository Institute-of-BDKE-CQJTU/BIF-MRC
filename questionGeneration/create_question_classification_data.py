import json
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')
questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']

def postfix_in_sentence(sentence):
    for i in range(6):
        if questionPostfix[i] in sentence:
            return i
    return -1

datas = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']

count = len(datas)
questionData = dict()
questionData['version'] = '1.0'
questionData['data'] = []

index_dict = dict(zip(map(str, list(range(6))), [0]*6))

postfix_dict = dict(zip(questionPostfix, list(range(6))))

for i in range(count):
    question = datas[i]['question']
    index = postfix_in_sentence(question)
    if index == -1:
        continue
    index_dict[str(index)] += 1
    newData = dict()
    newData['question'] = question
    answer = datas[i]['answer']
    newData['context'] = datas[i]['context']
    newData['answer'] = answer
    newData['postfix'] = questionPostfix[index]
    newData['prefix'] = question[:-len(questionPostfix[index])]
    answerLength = len(tokenizer.tokenize(answer))
    newData['answerLength'] = answerLength
    questionData['data'].append(newData)

count = len(questionData['data'])
type_data = [[], [], [], [], [], [], []]
for i in range(count):
    type_data[postfix_dict[questionData['data'][i]['postfix']]].append(questionData['data'][i])
print(type_data[0][0])

max_number = 0
for i in range(6):
    if index_dict[str(i)] > max_number:
        max_number = index_dict[str(i)]

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

questionData['data'].clear()
for i in range(max_number):
    for j in range(6):
        questionData['data'].append(type_data[j][i])

with open('./data/questionData/bridge/train.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(questionData,ensure_ascii=False))
 


