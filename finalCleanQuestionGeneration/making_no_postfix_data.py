import pickle
from transformers import BertTokenizer

f = open('./data/finalCleanQuestionGeneration/bridge/pretuning_train_data_6_3808.pkl', 'rb')
features = pickle.load(f)
f.close()

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

# ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
postfix_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in [['维', '修', '建', '议', '？'], ['是', '什', '么', '？'], ['是', '多', '少', '？'], 
                                                                        ['位', '于', '哪', '里','？'], ['的', '原', '因', '？'], ['问', '题', '？']]]

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

def detect_postfix(input_ids, token_type_ids):
    for i in range(3):
        try:
            postfix_start, postfix_end = findStartEnd(input_ids, postfix_ids[i], 0)
            if token_type_ids[postfix_end] == 0:
                return postfix_start, postfix_end-postfix_start+1
        except:
            continue
    return -1, 3

def find_sep_index(input_ids):
    flag = 0
    first_index = -1
    for i in range(len(input_ids)):
        if input_ids[i] == 102:
            if flag == 0:
                first_index = i
                flag = 1
            else:
                return first_index, i

new_features = []
for feature in features:
    input_ids = feature[0]
    attention_mask = feature[1]
    token_type_ids = feature[2]
    answer_start = feature[3]
    answer_end = feature[4]
    postfix_start, length = detect_postfix(input_ids, token_type_ids)
    if postfix_start == -1:
        new_features.append([input_ids, attention_mask, token_type_ids, answer_start, answer_end])
    else:
        input_ids = input_ids[:postfix_start] + input_ids[postfix_start+length:]+[0]*length
        first_index, second_index = find_sep_index(input_ids)
        attention_mask = [1]*(second_index+1) + [0]*(511-second_index)
        token_type_ids = [0]*(first_index+1) + [1]*(second_index - first_index) + [0]*(511 - second_index)
        assert len(token_type_ids) == 512
        assert len(attention_mask) == 512
        assert len(input_ids) == 512
        new_features.append([input_ids, attention_mask, token_type_ids, answer_start-length, answer_end-length])

print(len(new_features))
with open('./data/finalCleanQuestionGeneration/bridge/pretuning_train_data_3808_6_no_postfix.pkl', 'wb') as f:
    pickle.dump(new_features, f)
    