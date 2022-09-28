import random
from pyparsing import line
from transformers import BertTokenizer
import json
import pickle

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

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

def tokenizer_end_to_end_train_data():
    datas = json.load(open('./data/finalCleanQuestionGeneration/bridge/question_predict_train.json', 'r', encoding='UTF-8'))['data']
    # 这里获取数据
    # 三个后缀，带有负例数据
    postfix_three_with_other_index = [0, 4, 5, 6]
    postfix_three_with_other_data = []
    for data in datas:
        if data['postfix_index'] in postfix_three_with_other_index:
            postfix_three_with_other_data.append(data)
    
    # 这里tokenzier
    postfix_index = {0:0, 4:1, 5:2, 6:3}
    features = []
    for data in postfix_three_with_other_data:
        sentence = data['sentence']
        answer = data['answer']
        label = postfix_index[data['postfix_index']]
        sentence_tokens = tokenizer.tokenize(sentence)
        answer_tokens = tokenizer.tokenize(answer)
        # 如果答案太长直接丢掉
        if len(answer_tokens) > 256:
            break
        try:
            answer_tokens_start, answer_tokens_end = findStartEnd(sentence_tokens, answer_tokens, int(data['answer_start']))
        except:
            print(sentence, answer)
            print(sentence_tokens, answer_tokens)
            continue
        if len(sentence_tokens) > 254:
            # 如果sentence很长，但是答案在256内的，直接截断
            if answer_tokens_end < 254:
                sentence_tokens = sentence_tokens[:254]
            # 如果sentence很长，答案也在后面的
            else:
                sentence_exclude_answer_should_length = 254 - len(answer_tokens)
                sentence_exclude_answer_should_length_half = int(sentence_exclude_answer_should_length/2.0)
                if answer_tokens_start > sentence_exclude_answer_should_length_half:
                    sentence_tokens = sentence_tokens[answer_tokens_start - sentence_exclude_answer_should_length_half: answer_tokens_start - sentence_exclude_answer_should_length_half + 254]
                else:
                    sentence_tokens = sentence_tokens[:254]
        input_tokens = ['[CLS]']+sentence_tokens+['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_masks = [1]*len(input_ids)
        token_type_ids = [0]*(len(input_tokens))
        assert len(attention_masks) == len(token_type_ids)
        if len(attention_masks) < 256:
            length = 256 - len(attention_masks)
            for i in range(length):
                attention_masks.append(0)
                token_type_ids.append(0) 
                input_tokens.append('[PAD]')          
                input_ids.append(0)
        new_answer_start, new_answer_end = findStartEnd(input_tokens, answer_tokens, answer_tokens_start)
        features.append([input_ids, attention_masks, token_type_ids, label, new_answer_start, new_answer_end, answer, input_tokens])
        assert len(input_ids) == 256
    with open('data/finalCleanQuestionGeneration/bridge/postfix_three_with_other_data_end_to_end.pkl', 'wb') as f:
        pickle.dump(features, f)

def tokenizer_question_type_predict_train_data():
    datas = json.load(open('./data/finalCleanQuestionGeneration/bridge/question_predict_train.json', 'r', encoding='UTF-8'))['data']
    # 这里获取数据
    # 三个后缀，带有负例数据
    postfix_three_with_other_index = [0, 1, 2, 3, 4, 5, 6]
    postfix_three_with_other_data = []
    for data in datas:
        if data['postfix_index'] in postfix_three_with_other_index:
            postfix_three_with_other_data.append(data)
    
    # 这里tokenzier
    postfix_index = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6}
    features = []
    for data in postfix_three_with_other_data:
        sentence = data['sentence']
        label = postfix_index[data['postfix_index']]
        sentence_tokens = tokenizer.tokenize(sentence)
        # 如果文本太长直接丢掉后面的
        if len(sentence_tokens) > 254:
            sentence_tokens = sentence_tokens[:254]
        input_tokens = ['[CLS]']+sentence_tokens+['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_masks = [1]*len(input_ids)
        token_type_ids = [0]*(len(input_tokens))
        assert len(attention_masks) == len(token_type_ids)
        if len(attention_masks) < 256:
            length = 256 - len(attention_masks)
            for i in range(length):
                attention_masks.append(0)
                token_type_ids.append(0) 
                input_tokens.append('[PAD]')          
                input_ids.append(0)
        features.append([input_ids, attention_masks, token_type_ids, label])
    with open('data/finalCleanQuestionGeneration/bridge/question_type_predict_train_data_6.pkl', 'wb') as f:
        pickle.dump(features, f)

def tokenizer_question_type_predict_test_data():
    features = []
    with open('./data/finalCleanQuestionGeneration/bridge/clean.txt', 'r', encoding='UTF-8') as f:
        for line in f:
            templine = line.strip()
            line = templine.split('。')
            for sentence in line:
                if len(sentence) <= 10:
                    continue
                sentence_tokens = tokenizer.tokenize(sentence)
                if len(sentence_tokens) > 254:
                    sentence_tokens = sentence_tokens[:254]
                input_tokens = ['[CLS]']+sentence_tokens+['[SEP]']
                input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
                attention_masks = [1]*len(input_ids)
                token_type_ids = [0]*(len(input_tokens))
                assert len(attention_masks) == len(token_type_ids)
                if len(attention_masks) < 256:
                    length = 256 - len(attention_masks)
                    for i in range(length):
                        attention_masks.append(0)
                        token_type_ids.append(0) 
                        input_tokens.append('[PAD]')          
                        input_ids.append(0)
                features.append([input_ids, attention_masks, token_type_ids, sentence, templine])
    with open('data/finalCleanQuestionGeneration/bridge/question_type_predict_test_data.pkl', 'wb') as f:
        pickle.dump(features, f)

def tokenizer_answer_predict_train_data():
    datas = json.load(open('./data/finalCleanQuestionGeneration/bridge/question_predict_train.json', 'r', encoding='UTF-8'))['data']
    # 这里获取数据
    # 三个后缀，不带负例数据
    postfix_three_with_other_index = [0, 1, 2, 3, 4, 5]
    postfix_three_with_other_data = []
    for data in datas:
        if data['postfix_index'] in postfix_three_with_other_index:
            postfix_three_with_other_data.append(data)
    
    # 这里tokenzier
    postfix_index = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
    questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
    features = []
    for data in postfix_three_with_other_data:
        context = data['context']
        context_tokens = tokenizer.tokenize(context)
        postfix = questionPostfix[postfix_index[data['postfix_index']]]
        postfix_tokens = tokenizer.tokenize(postfix)
        answer = data['answer']
        answer_tokens = tokenizer.tokenize(answer)
        # answer太长直接不要
        if len(answer_tokens) > 254:
            continue
        try:
            answer_tokens_start, answer_tokens_end = findStartEnd(context_tokens, answer_tokens, int(data['answer_start']))
        except:
            continue
        if answer_tokens_end + len(postfix_tokens) + 2 > 254:
            # context tokens应该有的长度
            context_tokens_should_length = 253-len(postfix_tokens)
            # context tokens除去答案之后应有的长度
            context_tokens_exclude_answer_should_length = context_tokens_should_length - len(answer_tokens)
            context_tokens_exclude_answer_should_length_half = int(context_tokens_exclude_answer_should_length / 2)
            # 如果答案偏左
            if answer_tokens_start < context_tokens_exclude_answer_should_length_half:
                context_tokens = context_tokens[:context_tokens_should_length]
            else:
                context_tokens = context_tokens[answer_tokens_end - context_tokens_exclude_answer_should_length_half:answer_tokens_end - context_tokens_exclude_answer_should_length_half + context_tokens_should_length]
        else:
            if len(postfix_tokens) + 3+len(context_tokens) > 256:
                context_tokens = context_tokens[:253-len(postfix_tokens)]
        input_tokens = ['[CLS]']+postfix_tokens+['[SEP]']+context_tokens+['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        attention_masks = [1]*len(input_ids)
        token_type_ids = [0]*(len(postfix_tokens)+2) + [1]*(len(context_tokens)+1)
        assert len(attention_masks) == len(token_type_ids)
        if len(attention_masks) < 256:
            length = 256 - len(attention_masks)
            for i in range(length):
                attention_masks.append(0)
                token_type_ids.append(0) 
                input_tokens.append('[PAD]')          
                input_ids.append(0)
        try:
            new_answer_start, new_answer_end = findStartEnd(input_tokens, answer_tokens, answer_tokens_start+len(postfix_tokens)+2)
        except:
            continue
        features.append([input_ids, token_type_ids, attention_masks, new_answer_start, new_answer_end, input_tokens, answer])
        assert len(input_ids) == 256
    with open('data/finalCleanQuestionGeneration/bridge/answer_predict_train_data_6.pkl', 'wb') as f:
        pickle.dump(features, f)


def tokenizer_answer_precdict_test_data():
    f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_test_data.pkl', 'rb')
    features = pickle.load(f)
    f.close()
    sentences = [f[3] for f in features]
    lines = [f[4] for f in features]
    questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
    file=open('./finalCleanQuestionGeneration/question_postfix_6.txt','r')
    question_postfix_predict_str = file.readlines()[0].strip()[1:-1]
    question_postfix_predict = list(map(int, question_postfix_predict_str.split(',')))
    new_features_256 = []
    new_features_512 = []
    for i in range(len(lines)):
        if question_postfix_predict[i] == 6:
            continue
        line_tokens = tokenizer.tokenize(lines[i])
        sentence_tokens = tokenizer.tokenize(sentences[i])
        postfix = questionPostfix[question_postfix_predict[i]]
        postfix_tokens = tokenizer.tokenize(postfix)
        if len(sentence_tokens) +len(postfix_tokens) > 253:
            sentence_tokens = sentence_tokens[:253-len(postfix_tokens)]
        input_tokens_256 = ['[CLS]']+postfix_tokens+['[SEP]']+sentence_tokens+['[SEP]']
        if len(line_tokens) +len(postfix_tokens) > 509:
            line_tokens = line_tokens[:509-len(postfix_tokens)]
        input_tokens_512 = ['[CLS]']+postfix_tokens+['[SEP]']+line_tokens+['[SEP]']

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens_256)
        attention_masks = [1]*len(input_ids)
        token_type_ids = [0]*(len(input_tokens_256))
        assert len(attention_masks) == len(token_type_ids)
        if len(attention_masks) < 256:
            length = 256 - len(attention_masks)
            for j in range(length):
                attention_masks.append(0)
                token_type_ids.append(0) 
                input_tokens_256.append('[PAD]')          
                input_ids.append(0)
        assert len(input_ids) == 256
        new_features_256.append([input_ids, attention_masks, token_type_ids, input_tokens_256, sentences[i], lines[i]])

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens_512)
        attention_masks = [1]*len(input_ids)
        token_type_ids = [0]*(len(input_tokens_512))
        assert len(attention_masks) == len(token_type_ids)
        if len(attention_masks) < 512:
            length = 512 - len(attention_masks)
            for j in range(length):
                attention_masks.append(0)
                token_type_ids.append(0) 
                input_tokens_512.append('[PAD]')          
                input_ids.append(0)
        assert len(input_ids) == 512
        new_features_512.append([input_ids, attention_masks, token_type_ids, input_tokens_512, sentences[i], lines[i]])

    with open('data/finalCleanQuestionGeneration/bridge/answer_predict_test_256_6.pkl', 'wb') as f:
        pickle.dump(new_features_256, f)
    # with open('data/finalCleanQuestionGeneration/bridge/answer_predict_test_512_2.pkl', 'wb') as f:
    #     pickle.dump(new_features_512, f)

def tokenizer_question_type_random_for_answer_predict_test_data():
    f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_test_data.pkl', 'rb')
    features = pickle.load(f)
    f.close()
    sentences = [f[3] for f in features]
    lines = [f[4] for f in features]
    questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
    question_postfix_predict = []
    for i in range(len(lines)):
        question_postfix_predict.append(random.randint(0, 5))
    new_features_256 = []
    for i in range(len(lines)):
        sentence_tokens = tokenizer.tokenize(sentences[i])
        postfix = questionPostfix[question_postfix_predict[i]]
        postfix_tokens = tokenizer.tokenize(postfix)
        if len(sentence_tokens) +len(postfix_tokens) > 253:
            sentence_tokens = sentence_tokens[:253-len(postfix_tokens)]
        input_tokens_256 = ['[CLS]']+postfix_tokens+['[SEP]']+sentence_tokens+['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens_256)
        attention_masks = [1]*len(input_ids)
        token_type_ids = [0]*(len(input_tokens_256))
        assert len(attention_masks) == len(token_type_ids)
        if len(attention_masks) < 256:
            length = 256 - len(attention_masks)
            for j in range(length):
                attention_masks.append(0)
                token_type_ids.append(0) 
                input_tokens_256.append('[PAD]')          
                input_ids.append(0)
        assert len(input_ids) == 256
        new_features_256.append([input_ids, attention_masks, token_type_ids, input_tokens_256, sentences[i], lines[i]])
    print(len(new_features_256))
    with open('data/finalCleanQuestionGeneration/bridge/question_type_random_for_answer_predict_test_256_6.pkl', 'wb') as f:
        pickle.dump(new_features_256, f)

def temp_ensure_question_tokens(line_tokens, answer_start):
    j = 0
    for j in range(1, answer_start):
        if line_tokens[answer_start - j] in [',', '，', ':', '：', '.', '。', '、']:
            break
    question_prefix_tokens = line_tokens[answer_start - j+1:answer_start]
    return question_prefix_tokens

def tokenizer_answer_predict_test_data_random_answer():
    f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_test_data.pkl', 'rb')
    features = pickle.load(f)
    f.close()
    sentences = [f[3] for f in features]
    lines = [f[4] for f in features]
    questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
    file=open('./finalCleanQuestionGeneration/question_postfix_6.txt','r')
    question_postfix_predict_str = file.readlines()[0].strip()[1:-1]
    question_postfix_predict = list(map(int, question_postfix_predict_str.split(',')))
    new_features_512 = []
    for i in range(len(lines)):
        if question_postfix_predict[i] == 6:
            continue
        line_tokens = tokenizer.tokenize(lines[i])
        postfix = questionPostfix[question_postfix_predict[i]]
        postfix_tokens = tokenizer.tokenize(postfix)
        k = 0
        while k < 3: 
            answer_start = random.randint(0, len(line_tokens)-1)
            answer_end = random.randint(answer_start, len(line_tokens) - 1)
            question_prefix_tokens = temp_ensure_question_tokens(line_tokens, answer_start)
            question_tokens = question_prefix_tokens + postfix_tokens
            answer_start = answer_start + 2 + len(question_tokens)
            answer_end = answer_end + 2 + len(question_tokens)
            if answer_end < 510 and answer_start < 510:
                k += 1
            else:
                continue
            if len(line_tokens) +len(question_tokens) > 509:
                line_tokens = line_tokens[:509-len(postfix_tokens)]
            input_tokens_512 = ['[CLS]']+postfix_tokens+['[SEP]']+line_tokens+['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens_512)
            attention_masks = [1]*len(input_ids)
            token_type_ids = [0]*(len(input_tokens_512))
            assert len(attention_masks) == len(token_type_ids)
            if len(attention_masks) < 512:
                length = 512 - len(attention_masks)
                for j in range(length):
                    attention_masks.append(0)
                    token_type_ids.append(0) 
                    input_tokens_512.append('[PAD]')          
                    input_ids.append(0)
            assert len(input_ids) == 512
            new_features_512.append([input_ids, attention_masks, token_type_ids, answer_start, answer_end])
    print(len(new_features_512))
    with open('data/finalCleanQuestionGeneration/bridge/pretuning_train_data_random_answer_3808_6.pkl', 'wb') as f:
        pickle.dump(new_features_512[:3808], f)

def tokenizer_pretuning_train_data():
    f = open('./data/finalCleanQuestionGeneration/bridge/answer_predict_test_256.pkl', 'rb')
    features = pickle.load(f)
    all_input_ids = [f[0] for f in features]
    all_input_mask = [f[1] for f in features]
    all_label_ids = [f[2] for f in features]
    input_tokens = [f[3] for f in features]
    sentences = [f[4] for f in features]
    lines = [f[5] for f in features]
    file=open('./finalCleanQuestionGeneration/predict_start_index.txt','r')
    predict_start_index_str = file.readlines()[0].strip()[1:-1]
    predict_start_index = list(map(int, predict_start_index_str.split(',')))
    file.close()

    file=open('./finalCleanQuestionGeneration/predict_end_index.txt','r')
    predict_end_index_str = file.readlines()[0].strip()[1:-1]
    predict_end_index = list(map(int, predict_end_index_str.split(',')))
    file.close()
    f.close()

    answers_tokens = []
    for i in range(len(predict_end_index)):
        # print(input_tokens[i])
        answers_tokens.append(input_tokens[i][predict_start_index[i]:predict_end_index[i]+1])

    with open('./finalCleanQuestionGeneration/temp.txt', 'w', encoding='UTF-8') as f:
        for i in range(len(answers_tokens)):
            f.write(str(input_tokens[i]))
            f.write('\n')
            f.write(str(answers_tokens[i]))
            f.write('\n')
            f.write('='*50)
            f.write('\n')

def tokenizer_all_random_pretuning_data():
    f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_test_data.pkl', 'rb')
    features = pickle.load(f)
    f.close()
    sentences = [f[3] for f in features]
    lines = [f[4] for f in features]
    questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
    question_postfix_predict = []
    for i in range(len(lines)):
        question_postfix_predict.append(random.randint(0, 5))
    new_features_512 = []
    for i in range(len(lines)):
        if question_postfix_predict[i] == 6:
            continue
        line_tokens = tokenizer.tokenize(lines[i])
        postfix = questionPostfix[question_postfix_predict[i]]
        postfix_tokens = tokenizer.tokenize(postfix)
        k = 0
        while k < 3: 
            answer_start = random.randint(0, len(line_tokens)-1)
            answer_end = random.randint(answer_start, len(line_tokens) - 1)
            question_prefix_tokens = temp_ensure_question_tokens(line_tokens, answer_start)
            question_tokens = question_prefix_tokens + postfix_tokens
            answer_start = answer_start + 2 + len(question_tokens)
            answer_end = answer_end + 2 + len(question_tokens)
            if answer_end < 510 and answer_start < 510:
                k += 1
            else:
                continue
            if len(line_tokens) +len(question_tokens) > 509:
                line_tokens = line_tokens[:509-len(postfix_tokens)]
            input_tokens_512 = ['[CLS]']+postfix_tokens+['[SEP]']+line_tokens+['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(input_tokens_512)
            attention_masks = [1]*len(input_ids)
            token_type_ids = [0]*(len(input_tokens_512))
            assert len(attention_masks) == len(token_type_ids)
            if len(attention_masks) < 512:
                length = 512 - len(attention_masks)
                for j in range(length):
                    attention_masks.append(0)
                    token_type_ids.append(0) 
                    input_tokens_512.append('[PAD]')          
                    input_ids.append(0)
            assert len(input_ids) == 512
            new_features_512.append([input_ids, attention_masks, token_type_ids, answer_start, answer_end])
    print(len(new_features_512))
    with open('data/finalCleanQuestionGeneration/bridge/pretuning_train_data_all_random_3808_6.pkl', 'wb') as f:
        pickle.dump(new_features_512[:3808], f)

def tokenizer_random_data():
    datas = json.load(open('./data/finalCleanQuestionGeneration/bridge/construct_random_data_no_postfix.json', 'r', encoding='UTF-8'))['data']
    features = []
    for data in datas:
        question = data['question']
        context = data['context']
        answer = data['answer']
        input = tokenizer(question, context,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        questionTokenIds = tokenizer(question, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        start = LCS(tokenizer(context, return_tensors='pt')['input_ids'][0].tolist()[1:-1], questionTokenIds)
        answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        try:
            answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(questionTokenIds))
        except:
            continue
        input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()
        features.append([input_ids, attention_mask, token_type_ids, answerStart, answerEnd])
        if len(features) == 3350:
            write_data(features)
        if len(features) == 3808:
            write_data(features)
            break

def write_data(features):
    with open('data/finalCleanQuestionGeneration/bridge/random_data_'+str(len(features))+'_no_postfix.pkl', 'wb') as f:
        pickle.dump(features, f)
    # for i in range(len(sentences)):
    #     sentence_tokens = tokenizer.tokenize(sentences[i])
    #     line_tokens = tokenizer.tokenize(lines[i])


if __name__ == '__main__':
    # tokenizer_end_to_end_train_data()
    # tokenizer_question_type_predict_train_data()
    # tokenizer_question_type_predict_test_data()
    # tokenizer_answer_predict_train_data()
    # tokenizer_answer_precdict_test_data()
    # tokenizer_answer_predict_test_data_random_answer()
    # tokenizer_question_type_random_for_answer_predict_test_data()
    tokenizer_all_random_pretuning_data()
    # tokenizer_pretuning_train_data()
    # tokenizer_random_data()