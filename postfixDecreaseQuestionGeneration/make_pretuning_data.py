import json
import pickle
from transformers import BertTokenizer

file=open('./postfixDecreaseQuestionGeneration/predict_start_index.txt','r')
predict_start_index_str = file.readlines()[0].strip()[1:-1]
predict_start_index = list(map(int, predict_start_index_str.split(',')))
file.close()

file=open('./postfixDecreaseQuestionGeneration/predict_end_index.txt','r')
predict_end_index_str = file.readlines()[0].strip()[1:-1]
predict_end_index = list(map(int, predict_end_index_str.split(',')))
file.close()

datas = json.load(open('./data/preprocessingData/bridge/constructDomainTrainData4.json', 'r', encoding='UTF-8'))['data']
count = len(datas)
contexts, features = [], []
tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

questionPostfix = ['是多少？', '是什么？', '维修建议？']
file=open('./postfixDecreaseQuestionGeneration/question_postfix.txt','r')
question_postfix_predict_str = file.readlines()[0].strip()[1:-1]
question_postfix_predict = list(map(int, question_postfix_predict_str.split(',')))
postfixs = [questionPostfix[i] for i in question_postfix_predict]
file.close()
answers = []
for i in range(count):
    contexts.append(datas[i]['context'])
tokenizer_data = tokenizer(postfixs, contexts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
input_ids, attention_mask, token_type_ids = tokenizer_data['input_ids'].tolist(), tokenizer_data['attention_mask'].tolist(), tokenizer_data['token_type_ids'].tolist()
for i in range(len(contexts)):
    tokens = tokenizer.convert_ids_to_tokens(input_ids[i][predict_start_index[i]:predict_end_index[i]+1])
    predAnswer = ''.join(tokenizer.convert_tokens_to_string(tokens))
    predAnswer = predAnswer.replace(' ', '')
    answers.append(predAnswer)

newData = dict()
newData['version'] = '1.0'
newData['data'] = []
for i in range(len(contexts)):
    tempdata = dict()
    tempdata['context'] = contexts[i]
    tempdata['postfix'] = postfixs[i]
    tempdata['answer'] = answers[i]
    j = 0
    for j in range(len(contexts[i]) - len(answers[i])):
        if contexts[i][j:len(answers[i])+j] == answers[i]:
            break
    tempdata['answer_start'] = j
    newData['data'].append(tempdata)

with open('./data/postfixDecreaseQuestionGeneration/bridge/pretuning_data_train.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(newData,ensure_ascii=False))
# for i in range(len(contexts)):
#     features.append([input_ids[i], attention_mask[i], token_type_ids[i], predict_start_index[i], predict_end_index[i]])
# with open('./data/newQuestionGeneration/bridge/pretuning_data_train.pkl', 'wb') as f:
#     pickle.dump(features, f)