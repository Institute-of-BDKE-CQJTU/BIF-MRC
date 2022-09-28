from transformers import BertTokenizer
import json
import sys
import pickle
sys.path.append('./')
from utils import LCS, findStartEnd

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

# # 编码桥梁数据集
# # 编码训练集
# counts = [16, 32, 64, 128, 256, 512, 1024, 2048]
# for count in counts:
#     data = json.load(open('./data/sampleData/bridge/train-'+str(count)+'.json', 'r', encoding='UTF-8'))['data']
#     features = []
#     for i in range(count):
#         context = data[i]['context']
#         question = data[i]['question']
#         input = tokenizer(question, context,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#         answer = data[i]['answer']
#         questionTokenIds = tokenizer(question, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
#         start = LCS(tokenizer(context, return_tensors='pt')['input_ids'][0].tolist()[1:-1], questionTokenIds)
#         answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
#         try:
#             answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(questionTokenIds))
#         except:
#             print(count, i, answerTokenIds)
#             break

#         input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()

#         features.append([input_ids, attention_mask, token_type_ids, answerStart, answerEnd])

#     with open('./data/dataFeatures/bridge/train-'+str(count)+'.pkl', 'wb') as f:
#             pickle.dump(features, f)

seeds = [13, 43, 83, 181, 271, 347, 433, 659, 727, 859]
for seed in seeds:
    counts = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for count in counts:
        data = json.load(open('./data/sampleData/bridge/train-'+str(count)+'-'+str(seed)+'.json', 'r', encoding='UTF-8'))['data']
        features = []
        for i in range(count):
            context = data[i]['context']
            question = data[i]['question']
            input = tokenizer(question, context,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
            answer = data[i]['answer']
            questionTokenIds = tokenizer(question, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
            start = LCS(tokenizer(context, return_tensors='pt')['input_ids'][0].tolist()[1:-1], questionTokenIds)
            answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
            try:
                answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(questionTokenIds))
            except:
                print(count, i, answerTokenIds)
                break

            input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()

            features.append([input_ids, attention_mask, token_type_ids, answerStart, answerEnd])

        with open('./data/dataFeatures/bridge/train-'+str(count)+'-'+str(seed)+'.pkl', 'wb') as f:
                pickle.dump(features, f)

# # 编码验证集
# data = json.load(open('./data/preprocessingData/bridge/dev.json', 'r', encoding='UTF-8'))['data']
# features = []
# count = len(data)
# for i in range(count):
#     context = data[i]['context']
#     question = data[i]['question']
#     input = tokenizer(question, context,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#     trueAnswers = data[i]['answer']

#     input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()

#     features.append([input_ids, attention_mask, token_type_ids, trueAnswers])

# with open('./data/dataFeatures/bridge/dev.pkl', 'wb') as f:
#         pickle.dump(features, f)

# # 编码full_data
# data = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']
# features = []
# count = len(data)
# for i in range(count):
#     context = data[i]['context']
#     question = data[i]['question']
#     input = tokenizer(question, context,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#     answer = data[i]['answer']
#     questionTokenIds = tokenizer(question, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
#     start = LCS(tokenizer(context, return_tensors='pt')['input_ids'][0].tolist()[1:-1], questionTokenIds)
#     answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
#     try:
#         answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(questionTokenIds))
#     except:
#         print(count, i, answerTokenIds)
#         break

#     input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()

#     features.append([input_ids, attention_mask, token_type_ids, answerStart, answerEnd])

# with open('./data/dataFeatures/bridge/train.pkl', 'wb') as f:
#         pickle.dump(features, f)

# # 编码领域构造数据
# data = json.load(open('./data/preprocessingData/bridge/constructDomainTrainData4.json', 'r', encoding='UTF-8'))['data']
# features = []
# count = len(data)
# origin_index = []
# file=open('./test/all_drop_index.txt','r')
# drop_index_str = file.readlines()[0].strip()[1:-1]
# drop_index = list(map(int, drop_index_str.split(',')))
# file.close()
# questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
# question_type_number = [0, 0, 0, 0, 0, 0]
# lllll_index = []
# number = 0
# # 15126, 1419, 3851
# for i in range(count):
#     # if i in drop_index:
#     #     continue
#     context = data[i]['context']
#     question = data[i]['question']
#     j = 0
#     for j in range(6):
#         if questionPostfix[j] in question:
#             break
#     if j not in [0, 4]:
#         continue
#     # question = question[:-len(questionPostfix[j])]+'？'
#     input = tokenizer(question, context,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#     answer = data[i]['answer']
#     questionTokenIds = tokenizer(question, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
#     start = LCS(tokenizer(context, return_tensors='pt')['input_ids'][0].tolist()[1:-1], questionTokenIds)
#     answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
#     try:
#         answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(questionTokenIds))
#     except:
#         origin_index.append(i)
#         continue

#     if question_type_number[j] < 2129:
#         question_type_number[j] += 1
#     else:
#         continue
#     input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()
#     # question_type_number[j] += 1
#     features.append([input_ids, attention_mask, token_type_ids, answerStart, answerEnd])
#     lllll_index.append(i)
#     number += 1
#     # if number == 20189:
#     #     break
# # file=open('./test/origin_index2.txt','w')  
# # file.write(str(lllll_index))
# # file.close()
# print(len(features))
# print(question_type_number)
# with open('./data/postfixDecreaseQuestionGeneration/bridge/constructDomainTrainData_2129_postfix2.pkl', 'wb') as f:
#     pickle.dump(features, f)

# data = json.load(open('./data/preprocessingData/bridge/constructDomainTrainDataNoPostfix.json', 'r', encoding='UTF-8'))['data']
# features = []
# count = len(data)
# for i in range(count):
#     context = data[i]['context']
#     question = data[i]['question']
#     input = tokenizer(question, context,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
#     answer = data[i]['answer']
#     questionTokenIds = tokenizer(question, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
#     start = LCS(tokenizer(context, return_tensors='pt')['input_ids'][0].tolist()[1:-1], questionTokenIds)
#     answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
#     try:
#         answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(questionTokenIds))
#     except:
#         print(count, i, answerTokenIds)
#         continue

#     input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()

#     features.append([input_ids, attention_mask, token_type_ids, answerStart, answerEnd])

# with open('./data/dataFeatures/bridge/constructDomainTrainDataNoPostfix.pkl', 'wb') as f:
#         pickle.dump(features, f)