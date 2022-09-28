import json

datas = json.load(open('./data/preprocessingData/bridge/constructDomainTrainData.json', 'r', encoding='UTF-8'))['data']

newData = dict()
newData['version'] = '1.0'
newData['data'] = []

questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']

def wipe_out_postfix(question):
    for i in range(6):
        if questionPostfix[i] in question:
            return question[:-len(questionPostfix[i])]

for data in datas:
    data['question'] = wipe_out_postfix(data['question'])
    newData['data'].append(data)

with open('./data/preprocessingData/bridge/constructDomainTrainDataNoPostfix.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(newData,ensure_ascii=False))
