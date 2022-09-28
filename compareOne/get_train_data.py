from typing import List
from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

def find_same_span(input_tokens:List[str]):
    i, j = 0, 0
    same_location = []
    while i < len(input_tokens) - 1:
        j = i + 1
        while j < len(input_tokens):
            if input_tokens[j] == input_tokens[i]:
                k = 0
                for k in range(1, len(input_tokens) - j - 1):
                    if input_tokens[i+k] != input_tokens[j+k]:
                        break
                if k > 1:
                    temp_dict = dict()
                    temp_dict['first_start'] = i
                    temp_dict['first_end'] = i + k - 1 
                    temp_dict['second_start'] = j
                    temp_dict['second_end'] = j + k - 1
                    # temp_dict['span'] = input_tokens[i:i+k]
                    same_location.append(temp_dict)
                    i = i + k - 1 
                    break
            j += 1
        i += 1
    return same_location

def process_same_location(same_location, line_tokens):
    # 先去除被二次覆盖的span
    index = [0]*len(line_tokens)
    new_location = []
    for location in same_location:
        first_start = location['first_start']
        first_end = location['first_end']
        second_start = location['second_start']
        second_end = location['second_end']
        flag = 0
        for i in range(first_start, first_end+1):
            if index[i] == 1:
                flag = 1
                break
            else:
                index[i] = 1
        if flag == 0:
            for i in range(second_start, second_end+1):
                if index[i] == 1:
                    flag = 1
                    break
                else:
                    index[i] = 1
        if flag == 0:
            new_location.append(location)

    # 寻找相同的span
    final_data = dict()
    for location in new_location:
        span = line_tokens[location['first_start']:location['first_end']+1]
        span_str = ''.join(span)
        if span_str not in final_data.keys():
            final_data[span_str] = []
            final_data[span_str].append(str(location['first_start'])+'+'+str(location['first_end']))
            final_data[span_str].append(str(location['second_start'])+'+'+str(location['second_end']))
        else:
            if str(location['first_start'])+'+'+str(location['first_end']) not in final_data[span_str]:
                final_data[span_str].append(str(location['first_start'])+'+'+str(location['first_end']))
            if str(location['second_start'])+'+'+str(location['second_end']) not in final_data[span_str]:
                final_data[span_str].append(str(location['second_start'])+'+'+str(location['second_end']))
    return final_data


temp_dict = dict()
temp_dict['location_message'] = []
with open('./data/finalCleanQuestionGeneration/bridge/clean.txt', 'r', encoding='UTF-8') as f:
    for i, line in enumerate(f):
        index_dict = dict()
        index_dict['line_index'] = i
        templine = line.strip()
        line_tokens = tokenizer.tokenize(templine)
        index_dict['line'] = templine
        same_location = find_same_span(line_tokens)
        final_data = process_same_location(same_location, line_tokens)
        index_dict['location_data'] = final_data
        temp_dict['location_message'].append(index_dict)

with open('./data/compareOne/bridge/train_data.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(temp_dict,ensure_ascii=False))