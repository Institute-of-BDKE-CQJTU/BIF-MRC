import json

our_train_data_context = []
context = []
datas = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']
for data in datas:
    our_train_data_context.append(data['context'])

datas = json.load(open('./data/preprocessingData/bridge/dev.json', 'r', encoding='UTF-8'))['data']
for data in datas:
    our_train_data_context.append(data['context'])

datas = json.load(open('./data/preprocessingData/bridge/zong.json', 'r', encoding='UTF-8'))['data']
for data in datas:
    if data['context'] not in our_train_data_context:
        if data['context'] not in context:
            context.append(data['context'])

with open('./data/finalCleanQuestionGeneration/bridge/clean.txt', 'w', encoding='UTF-8') as f:
    for line in context:
        f.write(line+'\n')
