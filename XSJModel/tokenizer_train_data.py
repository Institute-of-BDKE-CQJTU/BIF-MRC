import json
from transformers import BertTokenizer
import pickle
from sklearn.model_selection import train_test_split
import numpy as np

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

def tokenizer_question_type():
    datas = json.load(open('./data/preprocessingData/bridge/zong_question_type.json', 'r', encoding='UTF-8'))['data']
    question, question_type = [], []
    features = []
    index_dict = {'before':0, 'middle':1, 'after':2}
    for data in datas:
        question.append(data['question'])
        question_type.append(index_dict[data['location']])

    tokenizer_data = tokenizer(question, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids, attention_mask, token_type_ids = tokenizer_data['input_ids'].tolist(), tokenizer_data['attention_mask'].tolist(), tokenizer_data['token_type_ids'].tolist()
    for i in range(len(question)):
        features.append([input_ids[i], attention_mask[i], token_type_ids[i], question_type[i]])
    with open('./data/XSJModel/bridge/question_type_train.pkl', 'wb') as f:
        pickle.dump(features, f)

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

def LCS(s1, s2):  
    """
        求最长公共子串，通过此算法定位多个相同答案的准确答案位置
    """ 
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  
    mmax=0   #最长匹配的长度  
    p=0  #最长匹配对应在s1中的最后一位  
    for i in range(len(s1)):  
        for j in range(len(s2)):  
            if s1[i]==s2[j]:  
                m[i+1][j+1]=m[i][j]+1  
                if m[i+1][j+1]>mmax:  
                    mmax=m[i+1][j+1]  
                    p=i+1  
    return p-mmax # 返回最长公共子串的起始位置

def sample_data():
    seeds = [43, 83, 271, 659, 859]
    datas = json.load(open('./data/preprocessingData/bridge/zong_question_type.json', 'r', encoding='UTF-8'))['data']
    length = len(datas)
    data_index = np.arange(0, length)
    new_data = dict()
    new_data['version'] = '1.0'
    new_data['data'] = []
    for seed in seeds:
        train_data, test_data = train_test_split(data_index, test_size=0.1, random_state=seed)
        train_data, test_data = train_data.tolist(), test_data.tolist()
        new_data['data'].clear()
        for i in train_data:
            new_data['data'].append(datas[i])
        with open('./data/XSJModel/bridge/train_'+str(seed)+'.json', 'w', encoding='UTF-8') as f:
            f.write(json.dumps(new_data,ensure_ascii=False))
        new_data['data'].clear()
        for i in test_data:
            new_data['data'].append(datas[i])
        with open('./data/XSJModel/bridge/test_'+str(seed)+'.json', 'w', encoding='UTF-8') as f:
            f.write(json.dumps(new_data,ensure_ascii=False))


def tokenizer_answer_predict_train_data():
    seeds = [43, 83, 271, 659, 859]
    index_dict = {'before':0, 'middle':1, 'after':2}
    for seed in seeds:
        datas = json.load(open('./data/XSJModel/bridge/train_'+str(seed)+'.json', 'r', encoding='UTF-8'))['data']
        features = []
        for data in datas:
            context = data['context']
            question = data['question']
            answer = data['answer']
            context_tokens = tokenizer.tokenize(context)
            question_tokens = tokenizer.tokenize(question)
            answer_tokens = tokenizer.tokenize(answer)
            # 如果答案都大于整个编码长度了，直接不要
            if len(answer_tokens) > 509 - len(question_tokens):
                continue
            try:
                answer_tokens_start, answer_tokens_end = findStartEnd(context_tokens, answer_tokens, int(data['answerStart']))
            except:
                print(context, answer)
                print(context_tokens)
                print(answer_tokens)
                return
            # answer_tokens_start, answer_tokens_end = findStartEnd(context_tokens, answer_tokens, int(data['answerStart']))
            if answer_tokens_end + len(question_tokens) + 2 > 510:
                # context tokens应该有的长度
                context_tokens_should_length = 509-len(question_tokens)
                # context tokens除去答案之后应有的长度
                context_tokens_exclude_answer_should_length = context_tokens_should_length - len(answer_tokens)
                context_tokens_exclude_answer_should_length_half = int(context_tokens_exclude_answer_should_length / 2)
                # 如果答案偏左
                if answer_tokens_start < context_tokens_exclude_answer_should_length_half:
                    context_tokens = context_tokens[:context_tokens_should_length]
                else:
                    context_tokens = context_tokens[answer_tokens_end - context_tokens_exclude_answer_should_length_half:answer_tokens_end - context_tokens_exclude_answer_should_length_half + context_tokens_should_length]
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
                for i in range(length):
                    attention_masks.append(0)
                    token_type_ids.append(0) 
                    input_tokens.append('[PAD]')          
                    input_ids.append(0)
            new_answer_start, new_answer_end = findStartEnd(input_tokens, answer_tokens, answer_tokens_start+len(question_tokens)+2)
            features.append([input_ids, token_type_ids, attention_masks, new_answer_start, new_answer_end, index_dict[data['location']]])
        print(features[0])
        with open('./data/XSJModel/bridge/train_'+str(seed)+'.pkl', 'wb') as f:
            pickle.dump(features, f)


def tokenizer_answer_predict_test_data(window_size=150):
    seeds = [43, 83, 271, 659, 859]
    index_dict = {'before':0, 'middle':1, 'after':2}
    for seed in seeds:
        datas = json.load(open('./data/XSJModel/bridge/test_'+str(seed)+'.json', 'r', encoding='UTF-8'))['data']
        features = []
        for index, data in enumerate(datas):
            context = data['context']
            question = data['question']
            answer = data['answer']
            context_tokens = tokenizer.tokenize(context)
            question_tokens = tokenizer.tokenize(question)
            context_tokens_list = []
            if len(context_tokens) > 509-len(question_tokens):
                # number 代表在当前滑动窗口大小下要取的次数
                context_should_max_length = 509 - len(question_tokens)
                number = int((len(context_tokens) - context_should_max_length) / window_size) + 1
                for i in range(number):
                    context_tokens_list.append(context_tokens[i*150:i*150+context_should_max_length])
            else:
                context_tokens_list.append(context_tokens)
            for i, context_tokens in enumerate(context_tokens_list):
                question_id = str(index)+'-'+str(i)
                input_tokens = ['[CLS]']+question_tokens+['[SEP]']+context_tokens+['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                attention_masks = [1]*len(input_ids)
                token_type_ids = [0]*(len(question_tokens)+2) + [1]*(len(context_tokens)+1)
                assert len(attention_masks) == len(token_type_ids)
                if len(attention_masks) < 512:
                    length = 512 - len(attention_masks)
                    for i in range(length):
                        attention_masks.append(0)
                        token_type_ids.append(0) 
                        input_tokens.append('[PAD]')          
                        input_ids.append(0)
                features.append([question_id, input_tokens, input_ids, token_type_ids, attention_masks, answer, index_dict[data['location']]])
                if index == 0:
                    print(features[0])
                    return 0
            # with open('./data/XSJModel/bridge/test_'+str(seed)+'.pkl', 'wb') as f:
            #     pickle.dump(features, f)


def test():
    print(findStartEnd([1, 2 ,3, 4, 5, 6, 7], [5, 6, 7], 0))

if __name__ == '__main__':
    # sample_data()
    # test()
    # tokenizer_answer_predict_train_data()
    tokenizer_answer_predict_test_data()
        