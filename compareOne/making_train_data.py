import json
from transformers import BertTokenizer
import pickle

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

datas = json.load(open('./data/compareOne/bridge/train_data.json', 'r', encoding='UTF-8'))['location_message']

features = []
count = 0
for data in datas:
    context = data['line']
    location_datas = data['location_data']
    context_tokens = tokenizer.tokenize(context)
    for span in location_datas:
        first_span_index = location_datas[span][0]
        second_span_index = location_datas[span][1]
        temp = first_span_index.split('+')
        first_span_start_index, first_span_end_index = int(temp[0]), int(temp[1])
        
        temp = second_span_index.split('+')
        second_span_start_index, second_span_end_index = int(temp[0]), int(temp[1])
        if second_span_end_index > 500:
            continue
        input_tokens = context_tokens[:second_span_start_index] + ['[QUESTION]'] + context_tokens[second_span_end_index+1:]
        if len(input_tokens) > 510:
            input_tokens = input_tokens[:510]
        input_tokens = ['[CLS]'] + input_tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1]*len(input_tokens)
        if len(attention_mask) < 512:
            length = 512 - len(attention_mask)
            for i in range(length):
                input_tokens.append('[PAD]')
                attention_mask.append(0)
                input_ids.append(0)
        features.append([input_ids, attention_mask, second_span_start_index+1, first_span_start_index+1, first_span_end_index+1])
        count += 1
        if count == 3808:
            break
    if count == 3808:
        break

with open('./data/compareOne/bridge/train.pkl', 'wb') as f:
    pickle.dump(features, f)