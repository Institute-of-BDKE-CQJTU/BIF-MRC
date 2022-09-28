from transformers import BertTokenizer
import json
import pickle

datas = json.load(open('./data/questionData/bridge/train.json', 'r', encoding='UTF-8'))['data']

count = len(datas)
prefixs = []
contexts = []
questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
postfix_to_index = dict(zip(questionPostfix, list(range(6))))
indexs = []
for i in range(count):
    prefixs.append(datas[i]['prefix'])
    contexts.append(datas[i]['context'])
    indexs.append(postfix_to_index[datas[i]['postfix']])

tokenizer = BertTokenizer.from_pretrained("./model/chinese_bert_wwm")

tokenizer_data = tokenizer(prefixs, contexts,padding='max_length', truncation=True, max_length=512, return_tensors='pt')

input_ids, attention_mask, token_type_ids = tokenizer_data['input_ids'].tolist(), tokenizer_data['attention_mask'].tolist(), tokenizer_data['token_type_ids'].tolist()

features = []
for i in range(count):
    features.append([input_ids[i], attention_mask[i], token_type_ids[i], indexs[i]])

with open('./data/questionData/bridge/train.pkl', 'wb') as f:
    pickle.dump(features, f)