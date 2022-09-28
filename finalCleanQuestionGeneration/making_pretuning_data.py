import json
import pickle
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

file=open('./finalCleanQuestionGeneration/predict_start_index_256_6.txt','r')
predict_start_index_str = file.readlines()[0].strip()[1:-1]
predict_start_index = list(map(int, predict_start_index_str.split(',')))
file.close()

file=open('./finalCleanQuestionGeneration/predict_end_index_256_6.txt','r')
predict_end_index_str = file.readlines()[0].strip()[1:-1]
predict_end_index = list(map(int, predict_end_index_str.split(',')))
file.close()

f = open('./data/finalCleanQuestionGeneration/bridge/question_type_random_for_answer_predict_test_256_6.pkl', 'rb')
features = pickle.load(f)
f.close()
inputs_tokens = [f[3] for f in features]
sentences = [f[4] for f in features]
lines = [f[5] for f in features]

def ensure_question_tokens(input_tokens, answer_start):
    postfix_tokens = None
    for i in range(1, 7):
        if input_tokens[i] == '[SEP]':
            postfix_tokens = input_tokens[1:i]
    flag = 0
    if '建' in postfix_tokens and '议' in postfix_tokens:
        flag = 1

    for i in range(1, 5):
        if input_tokens[answer_start - i] == '：':
            for j in range(1, answer_start - i):
                if input_tokens[answer_start - i - j] == '[SEP]' or input_tokens[answer_start - i - j] in [',', '，', '。', '、', '.']:
                    if flag == 1 and '建' in input_tokens[answer_start - i - j + 1:answer_start-i] and '议' in input_tokens[answer_start - i - j + 1:answer_start-i]:
                        return input_tokens[answer_start - i - j + 1:answer_start-i] + ['？']
                    else:
                        return input_tokens[answer_start - i - j + 1:answer_start-i] + postfix_tokens

    if input_tokens[answer_start - 1] == '为' or input_tokens[answer_start - 1] == '，':
        for i in range(2, answer_start):
            if input_tokens[answer_start - i] in [',', '，', '：', '。', '、', '.'] or input_tokens[answer_start - i] == '[SEP]':
                if flag == 1 and '建' in input_tokens[answer_start - i + 1:answer_start-1] and '议' in input_tokens[answer_start - i + 1:answer_start-1]:
                    return input_tokens[answer_start - i + 1:answer_start-1] + ['？']
                else:
                    return input_tokens[answer_start - i + 1:answer_start-1] + postfix_tokens
                
    if input_tokens[answer_start - 1] == '）':
        if input_tokens[answer_start - 3] == '（':
            for j in range(1, answer_start - 3):
                if input_tokens[answer_start - 3 - j] == '[SEP]' or input_tokens[answer_start - 3 - j] in [',', '，', '。', '、', '.']:
                    if flag == 1 and '建' in input_tokens[answer_start - 3 - j + 1:answer_start-3] and '议' in input_tokens[answer_start - 3 - j + 1:answer_start-3]:
                        return input_tokens[answer_start - 3 - j + 1:answer_start-3] + ['？']
                    else:
                        return input_tokens[answer_start - 3 - j + 1:answer_start-3] + postfix_tokens
    for i in range(1, answer_start):
        if input_tokens[answer_start - i] == '[SEP]' or input_tokens[answer_start - i] in [',', '，', '。', '、', '.']:
            if flag == 1 and '建' in input_tokens[answer_start - i + 1:answer_start] and '议' in input_tokens[answer_start - i + 1:answer_start]:
                return input_tokens[answer_start - i + 1:answer_start] + ['？']
            else:
                return input_tokens[answer_start - i + 1:answer_start] + postfix_tokens

def findStartEnd(paragraphTokensList, answerTokensList, strAnswerStart):
    """
        从paragraph tokens中寻找answer tokens 的起始和终止位置
        针对多个位置，取其与strAnswerStart最短的位置
        返回的是下标
        paragraphTokensList：list
        answerTokensList：list
        strAnswerStart:int
    """
    positions = []
    for i in range(len(paragraphTokensList)-len(answerTokensList)+1):
        if paragraphTokensList[i] == answerTokensList[0]:
            flag = True
            for j in range(1, len(answerTokensList)):
                if paragraphTokensList[i+j] != answerTokensList[j]:
                    flag = False
                    break
            if flag == True:
                positions.append([i, i+len(answerTokensList)-1])
    
    if len(positions) > 1:
        minDistance = 1000
        index = -1
        for i, position in enumerate(positions):
            if abs(position[0] - strAnswerStart) < minDistance:
                minDistance = abs(position[0] - strAnswerStart)
                index = i
        return positions[index][0], positions[index][1]
    else:
        return positions[0][0], positions[0][1]

# 看生成的问题除了后缀之外短不短
def short_question_without_postfix_length(question_tokens):
    if len(question_tokens) < 5:
        return True
    length = None
    for i in range(len(question_tokens)-1):
        if question_tokens[i] == '维' and question_tokens[i+1] == '修':
            length = len(question_tokens) - 5
            break
        if question_tokens[i] == '是' and question_tokens[i+1] == '什':
            length = len(question_tokens) - 4
            break
        if question_tokens[i] == '是' and question_tokens[i+1] == '多':
            length = len(question_tokens) - 4
            break
        if question_tokens[i] == '位' and question_tokens[i+1] == '于':
            length = len(question_tokens) - 5
            break
    if length is None:
        for i in range(len(question_tokens)-1):
            if question_tokens[i] == '建' and question_tokens[i+1] == '议':
                return False
    if question_tokens[0] == '长' and question_tokens[1] == '是':
        return False
    if length is None or length <= 1:
        return True
    else:
        return False        

# 如果第一次生成的问题太短，二次生成
def second_process_question(line_tokens, answer_start):
    pass

def process_suggestion_question(answer_tokens):
    if answer_tokens[0] == '对':
        for i in range(len(answer_tokens) - 2):
            if answer_tokens[i] == '处':
                question_tokens = answer_tokens[:i+1]+['维', '修', '建', '议', '？']
                answer_tokens = answer_tokens[i+1:]
                return question_tokens, answer_tokens
            if answer_tokens[i] == '进' and answer_tokens[i+1] == '行':
                question_tokens = answer_tokens[:i]+['维', '修', '建', '议', '？']
                answer_tokens = answer_tokens[i:]
                return question_tokens, answer_tokens
    if answer_tokens[0] == '针' and answer_tokens[1] == '对':
        for i in range(len(answer_tokens) - 1):
            if answer_tokens[i] == '情' and answer_tokens[i+1] == '况':
                question_tokens = answer_tokens[:i+2]+['维', '修', '建', '议', '？']
                answer_tokens = answer_tokens[i+2:]
                return question_tokens, answer_tokens
    return 1, 1
new_features = []

def temp_ensure_question_tokens(input_tokens, answer_start):
    postfix_tokens = None
    for i in range(1, 7):
        if input_tokens[i] == '[SEP]':
            postfix_tokens = input_tokens[1:i]
    j = 0
    for j in range(1, answer_start):
        if input_tokens[answer_start - j] in [',', '，', ':', '：', '.', '。', '、']:
            break
    question_tokens = input_tokens[answer_start - j+1:answer_start]+postfix_tokens
    return question_tokens

for i in range(len(predict_end_index)):
    # if predict_end_index[i] <= predict_start_index[i]:
    #     continue
    # if '[SEP]' in inputs_tokens[i][predict_start_index[i]:predict_end_index[i]+1]:
    #     continue
    # question_tokens = ensure_question_tokens(inputs_tokens[i], predict_start_index[i])
    # if question_tokens is None:
    #     continue
    # if '；' in question_tokens:
    #     for j in range(len(question_tokens)):
    #         if question_tokens[j] == '；':
    #             question_tokens = question_tokens[j+1:]
    #             break
    # if '）' in question_tokens and '（' in question_tokens:
    #     for j in range(len(question_tokens)):
    #         if question_tokens[j] == '）':
    #             question_tokens = question_tokens[j+1:]
    #             break
    # if len(question_tokens) >= 2 and question_tokens[1] == '）':
    #     question_tokens = question_tokens[2:]
    # context_tokens = tokenizer.tokenize(lines[i])
    # answer_tokens = inputs_tokens[i][predict_start_index[i]:predict_end_index[i]+1]
    # if question_tokens == ['维', '修', '建', '议', '？']:
    #     A, B = process_suggestion_question(answer_tokens)
    #     if type(A) != int:
    #         question_tokens, answer_tokens = A, B
    # if short_question_without_postfix_length(question_tokens):
    #     continue
    question_tokens = temp_ensure_question_tokens(inputs_tokens[i], predict_start_index[i])
    context_tokens = tokenizer.tokenize(lines[i])
    answer_tokens = inputs_tokens[i][predict_start_index[i]:predict_end_index[i]+1]
    try:
        answer_start, answer_end = findStartEnd(context_tokens, answer_tokens, 0)
    except:
        continue
    if answer_end + len(question_tokens) + 2 > 510:
        # context tokens应该有的长度
        context_tokens_should_length = 509-len(question_tokens)
        # context tokens除去答案之后应有的长度
        context_tokens_exclude_answer_should_length = context_tokens_should_length - len(answer_tokens)
        context_tokens_exclude_answer_should_length_half = int(context_tokens_exclude_answer_should_length / 2)
        # 如果答案偏左
        if answer_start < context_tokens_exclude_answer_should_length_half:
            context_tokens = context_tokens[:context_tokens_should_length]
        else:
            context_tokens = context_tokens[answer_end - context_tokens_exclude_answer_should_length_half:answer_end - context_tokens_exclude_answer_should_length_half + context_tokens_should_length]
    else:
        if len(question_tokens) + 3+len(context_tokens) > 512:
            context_tokens = context_tokens[:509-len(question_tokens)]
    input_tokens = ['[CLS]']+question_tokens+['[SEP]']+context_tokens+['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    attention_masks = [1]*len(input_ids)
    token_type_ids = [0]*(len(question_tokens)+2) + [1]*(len(context_tokens)+1)
    assert len(attention_masks) == len(token_type_ids)
    if len(attention_masks) < 512:
        length = 512 - len(attention_masks)
        for j in range(length):
            attention_masks.append(0)
            token_type_ids.append(0) 
            input_tokens.append('[PAD]')          
            input_ids.append(0)
    new_answer_start, new_answer_end = findStartEnd(input_tokens, answer_tokens, answer_start+len(question_tokens)+2)
    new_features.append([input_ids, attention_masks, token_type_ids, new_answer_start, new_answer_end])
    assert len(input_ids) == 512
    if len(new_features) == 3808:
        break
count = len(new_features)
print(count)
with open('./data/finalCleanQuestionGeneration/bridge/question_type_random_pretuning_train_data_6_new1.pkl', 'wb') as f:
    pickle.dump(new_features, f)
