import json
from transformers import BertTokenizer
import random
import pickle

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

datas = json.load(open('./data/compareTwo/bridge/QA_pair.json', 'r', encoding='UTF-8'))['data']

contexts = []
with open('./data/finalCleanQuestionGeneration/bridge/clean.txt', 'r', encoding='UTF-8') as f:
    for line in f:
        line = line.strip()
        contexts.append(line)

contexts = contexts + contexts + contexts

QA_pair = []
for data in datas:
    QA_pair.append([data['question'], data['answer']])

features = []
for i in range(len(contexts)):
    context_tokens = tokenizer.tokenize(contexts[i])
    question_tokens = tokenizer.tokenize(QA_pair[i][0])
    answer_tokens = tokenizer.tokenize(QA_pair[i][1])
    if len(context_tokens) + len(question_tokens) + len(answer_tokens) + 3 > 512:
        context_tokens = context_tokens[:509 - len(question_tokens) - len(answer_tokens)]
    input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + context_tokens + ['[SEP]']
    answer_start = random.randint(len(question_tokens)+2, len(input_tokens) - 1)
    input_tokens = input_tokens[:answer_start] + answer_tokens + input_tokens[answer_start:]
    attention_mask = [1]*len(input_tokens)
    token_type_ids = [0]*(len(question_tokens) + 2) + [1]*(len(input_tokens) - 2 - len(question_tokens))
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    if len(input_tokens) < 512:
        length = 512 - len(input_ids)
        for _ in range(length):
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)
            input_tokens.append('[PAD]')
    assert len(input_ids) == 512
    features.append([input_ids, attention_mask, token_type_ids, answer_start, answer_start+len(answer_tokens)])

print(len(features))
with open('data/compareTwo/bridge/pretuning_train_data.pkl', 'wb') as f:
    pickle.dump(features[:3808], f)
