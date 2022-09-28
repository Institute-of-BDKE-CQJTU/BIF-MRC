import json


# 预处理桥梁数据集
# # 预处理训练集
# newData = dict()
# newData['version'] = '1.0'
# newData['data'] = []


# originData = json.load(open('./data/rawData/bridge/BS_train.json', 'r', encoding='UTF-8'))['data']

# for article in originData:
#     paragraphs = article['paragraphs']
#     for paragraph in paragraphs:
#         qas = paragraph['qas']
#         for qa in qas:
#             pairData = dict()
#             pairData['id'] = qa['id']
#             pairData['context'] = paragraph['context']
#             pairData['question'] = qa['question']
#             pairData['answer'] = qa['answers'][0]['text']
#             pairData['answerStart'] = qa['answers'][0]['answer_start']
#             newData['data'].append(pairData)

# with open('./data/preprocessingData/bridge/train.json', 'w', encoding='UTF-8') as f:
#     f.write(json.dumps(newData,ensure_ascii=False))

# # 预处理验证集
# newData = dict()
# newData['version'] = '1.0'
# newData['data'] = []

# originData = json.load(open('./data/rawData/bridge/BS_dev.json', 'r', encoding='UTF-8'))['data']

# for article in originData:
#     paragraphs = article['paragraphs']
#     for paragraph in paragraphs:
#         qas = paragraph['qas']
#         for qa in qas:
#             pairData = dict()
#             pairData['id'] = qa['id']
#             pairData['context'] = paragraph['context']
#             pairData['question'] = qa['question']
#             pairData['answer'] = []
#             pairData['answer'].append(qa['answers'][0]['text'])
#             pairData['answerStart'] = qa['answers'][0]['answer_start']
#             newData['data'].append(pairData)

# with open('./data/preprocessingData/bridge/dev.json', 'w', encoding='UTF-8') as f:
#     f.write(json.dumps(newData,ensure_ascii=False))

# 预处理验证集
newData = dict()
newData['version'] = '1.0'
newData['data'] = []

originData = json.load(open('./data/rawData/bridge/zong_dev.json', 'r', encoding='UTF-8'))['data']

for article in originData:
    paragraphs = article['paragraphs']
    for paragraph in paragraphs:
        qas = paragraph['qas']
        for qa in qas:
            pairData = dict()
            pairData['id'] = qa['id']
            pairData['context'] = paragraph['context']
            pairData['question'] = qa['question']
            pairData['answer'] = qa['answers'][0]['text']
            pairData['answerStart'] = qa['answers'][0]['answer_start']
            newData['data'].append(pairData)

originData = json.load(open('./data/rawData/bridge/zong_train.json', 'r', encoding='UTF-8'))['data']

for article in originData:
    paragraphs = article['paragraphs']
    for paragraph in paragraphs:
        qas = paragraph['qas']
        for qa in qas:
            pairData = dict()
            pairData['id'] = qa['id']
            pairData['context'] = paragraph['context']
            pairData['question'] = qa['question']
            pairData['answer'] = qa['answers'][0]['text']
            pairData['answerStart'] = qa['answers'][0]['answer_start']
            newData['data'].append(pairData)


with open('./data/preprocessingData/bridge/zong.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(newData,ensure_ascii=False))