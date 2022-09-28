from transformers import BertTokenizer
import json
import sys
import pickle
sys.path.append('./')
from prepareData.utils import LCS, findStartEnd

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')


seeds = [13, 181, 347, 433, 727]
for seed in seeds:
    counts = [16, 32, 64, 128, 256, 512, 1024, 2048]
    for count in counts:
        data = json.load(open('./data/sampleData/bridge/train-'+str(count)+'-'+str(seed)+'.json', 'r', encoding='UTF-8'))['data']
        features = []
        for i in range(count):
            context = data[i]['context']
            question = data[i]['question']
            answer = data[i]['answer']
            context_tokens = tokenizer.tokenize(context)
            question_tokens = tokenizer.tokenize(question)
            answer_tokens = tokenizer.tokenize(answer)
            answer_str_location = data[i]['answerStart']
            try:
                answer_start, answer_end = findStartEnd(context_tokens, answer_tokens, answer_str_location)
            except:
                continue
            if answer_end + len(question_tokens) > 508:
                context_tokens = context_tokens[answer_end-(508-len(question_tokens)):answer_end]
            else:
                if len(context_tokens) + len(question_tokens) > 508:
                    context_tokens = context_tokens[:508-len(question_tokens)]
            input_tokens = ['[CLS]']+context_tokens+['[SEP]'] + question_tokens + ['[QUESTION]'] + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
            attention_mask = [1]*len(input_ids)
            if len(input_ids) < 512:
                length = 512 - len(input_ids)
                for _ in range(length):
                    input_tokens.append('[PAD]')
                    input_ids.append(0)
                    attention_mask.append(0)  
            try:
                new_answer_start, new_answer_end = findStartEnd(input_tokens, answer_tokens, answer_start)
            except:
                continue
            features.append([input_ids, attention_mask, len(context_tokens)+len(question_tokens)+2, new_answer_start, new_answer_end])
        with open('./data/compareOne/bridge/train-'+str(count)+'-'+str(seed)+'.pkl', 'wb') as f:
                pickle.dump(features, f)

# data = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']
# features = []
# count = len(data)
# for i in range(count):
#     context = data[i]['context']
#     question = data[i]['question']
#     answer = data[i]['answer']
#     context_tokens = tokenizer.tokenize(context)
#     question_tokens = tokenizer.tokenize(question)
#     answer_tokens = tokenizer.tokenize(answer)
#     answer_str_location = data[i]['answerStart']
#     try:
#         answer_start, answer_end = findStartEnd(context_tokens, answer_tokens, answer_str_location)
#     except:
#         continue
#     if answer_end + len(question_tokens) > 508:
#         context_tokens = context_tokens[answer_end-(508-len(question_tokens)):answer_end]
#     else:
#         if len(context_tokens) + len(question_tokens) > 508:
#             context_tokens = context_tokens[:508-len(question_tokens)]
#     input_tokens = ['[CLS]']+context_tokens+['[SEP]'] + question_tokens + ['[QUESTION]'] + ['[SEP]']
#     input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
#     attention_mask = [1]*len(input_ids)
#     if len(input_ids) < 512:
#         length = 512 - len(input_ids)
#         for _ in range(length):
#             input_tokens.append('[PAD]')
#             input_ids.append(0)
#             attention_mask.append(0)  
#     try:
#         new_answer_start, new_answer_end = findStartEnd(input_tokens, answer_tokens, answer_start)
#     except:
#         continue
#     features.append([input_ids, attention_mask, len(context_tokens)+len(question_tokens)+2, new_answer_start, new_answer_end])
# with open('./data/compareOne/bridge/train.pkl', 'wb') as f:
#     pickle.dump(features, f)


# data = json.load(open('./data/preprocessingData/bridge/dev.json', 'r', encoding='UTF-8'))['data']
# features = []
# count = len(data)
# for i in range(count):
#     context = data[i]['context']
#     question = data[i]['question']
#     answer = data[i]['answer'][0]
#     context_tokens = tokenizer.tokenize(context)
#     question_tokens = tokenizer.tokenize(question)
#     if len(context_tokens) + len(question_tokens) > 508:
#         context_tokens = context_tokens[:508-len(question_tokens)]
#     input_tokens = ['[CLS]']+context_tokens+['[SEP]'] + question_tokens + ['[QUESTION]'] + ['[SEP]']
#     input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
#     attention_mask = [1]*len(input_ids)
#     if len(input_ids) < 512:
#         length = 512 - len(input_ids)
#         for _ in range(length):
#             input_tokens.append('[PAD]')
#             input_ids.append(0)
#             attention_mask.append(0)  
#     features.append([input_ids, attention_mask, len(context_tokens)+len(question_tokens)+2, input_tokens, answer])
# with open('./data/compareOne/bridge/dev.pkl', 'wb') as f:
#     pickle.dump(features, f)