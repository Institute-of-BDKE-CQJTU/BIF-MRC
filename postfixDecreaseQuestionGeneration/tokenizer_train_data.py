from transformers import BertTokenizer
import json
import sys
import pickle


def findStartEnd(paragraphTokensList, answerTokensList, strAnswerStart):
    """
        从paragraph tokens中寻找answer tokens 的起始和终止位置
        针对多个位置，取其与strAnswerStart最短的位置
        paragraphTokensList：list
        answerTokensList：list
        strAnswerStart:int
    """
    positions = []
    for i in range(len(paragraphTokensList)):
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

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

datas = json.load(open('./data/postfixDecreaseQuestionGeneration/bridge/question_predict_train.json', 'r', encoding='UTF-8'))['data']
count = len(datas)
print(count)
newData = dict()
newData['version'] = '1.0'
newData['data'] = []   

def remove_no_answer():
    for i in range(count):
        sentence = datas[i]['context']
        answer = datas[i]['answer']
        sentenceTokenIds = tokenizer(sentence, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        try:
            answerStart, answerEnd = findStartEnd(sentenceTokenIds, answerTokenIds, datas[i]['answer_start'])
            newData['data'].append(datas[i])
        except:
            continue
    with open('./data/postfixDecreaseQuestionGeneration/bridge/question_predict_train.json', 'w', encoding='UTF-8') as f:
        f.write(json.dumps(newData,ensure_ascii=False))

def tokenize_question_type():
    features, sentences, postfix_index = [], [], []
    for i in range(count):
        sentences.append(datas[i]['context'])
        postfix_index.append(int(datas[i]['postfix_index']))
    tokenizer_data = tokenizer(sentences, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids, attention_mask, token_type_ids = tokenizer_data['input_ids'].tolist(), tokenizer_data['attention_mask'].tolist(), tokenizer_data['token_type_ids'].tolist()
    for i in range(len(sentences)):
        features.append([input_ids[i], attention_mask[i], token_type_ids[i], postfix_index[i]])
    with open('./data/postfixDecreaseQuestionGeneration/bridge/question_type_train.pkl', 'wb') as f:
        pickle.dump(features, f)

def tokenizer_answer_predict():
    features = []
    for i in range(count):
        sentence = datas[i]['context']
        postfix = datas[i]['postfix']
        input = tokenizer(postfix, sentence,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        answer = datas[i]['answer']
        postfixTokenIds = tokenizer(postfix, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        start = LCS(tokenizer(sentence, return_tensors='pt')['input_ids'][0].tolist()[1:-1], postfixTokenIds)
        answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        try:
            answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(postfixTokenIds))
        except:
            print(count, i, answerTokenIds)
            continue

        input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()

        features.append([input_ids, attention_mask, token_type_ids, answerStart, answerEnd])
    with open('./data/postfixDecreaseQuestionGeneration/bridge/answer_predict_train.pkl', 'wb') as f:
        pickle.dump(features, f)

def tokenizer_end_to_end():
    features, postfix_index = [], None
    for i in range(count):
        sentence = datas[i]['context']
        postfix = datas[i]['postfix']
        if postfix == 'negative':
            continue
        input = tokenizer(postfix, sentence,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        answer = datas[i]['answer']
        postfixTokenIds = tokenizer(postfix, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        start = LCS(tokenizer(sentence, return_tensors='pt')['input_ids'][0].tolist()[1:-1], postfixTokenIds)
        answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        try:
            answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(postfixTokenIds))
            postfix_index = datas[i]['postfix_index']
        except:
            print(count, i, answerTokenIds)
            continue

        input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()

        features.append([input_ids, attention_mask, token_type_ids, postfix_index, answerStart, answerEnd])
    with open('./data/newQuestionGeneration/bridge/end_to_end_train.pkl', 'wb') as f:
        pickle.dump(features, f)

def tokenizer_pretuning_train_data():
    questionPostfix = ['是多少？', '是什么？', '维修建议？']
    postfix_to_index = dict(zip(questionPostfix, list(range(6))))
    question_type_number = [0, 0, 0, 0, 0, 0]
    datas = json.load(open('./data/postfixDecreaseQuestionGeneration/bridge/pretuning_data_train.json', 'r', encoding='UTF-8'))['data']
    count = len(datas)
    features = []
    print(count)
    # file=open('./test/all_drop_index.txt','r')
    # drop_index_str = file.readlines()[0].strip()[1:-1]
    # drop_index = list(map(int, drop_index_str.split(',')))
    # file.close()
    newData = dict()
    newData['version'] = '1.0'
    newData['data'] = []
    temp_index = []
    lllll_index = []
    question_length = []
    for i in range(count):
        # if i in drop_index:
        #     continue
        context = datas[i]['context']
        answer_start = int(datas[i]['answer_start'])
        j = 0
        for j in range(1, answer_start):
            if context[answer_start - j] in [',', '，', ':', '：', '.', '。', '、']:
                break
        postfix = datas[i]['postfix']
        # 要后缀
        # question = context[answer_start - j+1:answer_start]+postfix
        # 不要后缀
        question = context[answer_start - j+1:answer_start]+'？'
        # question = context[:answer_start]+postfix
        if len(question) < 8:
            temp_index.append(i)
            continue
        if answer_start > 250:
            if answer_start+250 > len(context):
                context = context[answer_start-250:]
            else:
                context = context[answer_start-250:answer_start+250]
        answer = datas[i]['answer']
        if len(answer) == 0:
            temp_index.append(i)
            continue 
        input = tokenizer(question, context, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        questionTokenIds = tokenizer(question, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        start = LCS(tokenizer(context, return_tensors='pt')['input_ids'][0].tolist()[1:-1], questionTokenIds)
        answerTokenIds = tokenizer(answer, return_tensors='pt')['input_ids'][0].tolist()[1:-1]
        try:
            answerStart, answerEnd = findStartEnd(input['input_ids'][0].tolist(), answerTokenIds, start+len(questionTokenIds))
        except:
            tempdata = dict()
            tempdata['context'] = context
            tempdata['question'] = question
            tempdata['answer'] = answer
            newData['data'].append(tempdata)
            temp_index.append(i)
            continue
        if question_type_number[postfix_to_index[postfix]] < 1419:
            question_type_number[postfix_to_index[postfix]] += 1
        else:
            continue
        input_ids, attention_mask, token_type_ids = input['input_ids'][0].tolist(), input['attention_mask'][0].tolist(), input['token_type_ids'][0].tolist()
        # question_type_number[postfix_to_index[postfix]] += 1
        # if question_type_number[0] > 14623 and postfix_to_index[postfix] == 0:
        #     continue
        
        features.append([input_ids, attention_mask, token_type_ids, answerStart, answerEnd])

        lllll_index.append(i)
        question_length.append(len(question))
    # file=open('./test/pretuning_index2.txt','w')  
    # file.write(str(lllll_index))
    # file.close()
    # file=open('./test/pretuning_question_length.txt','w')  
    # file.write(str(question_length))
    # file.close()
    print(len(features))
    print(question_type_number)
    with open('./data/postfixDecreaseQuestionGeneration/bridge/pretuning_data_train_new_1419_nopostfix.pkl', 'wb') as f:
        pickle.dump(features, f)

if __name__ == "__main__":
    # 这个函数只能执行一次
    # remove_no_answer()
    # tokenize_question_type()
    # tokenizer_answer_predict()
    # tokenizer_end_to_end()
    tokenizer_pretuning_train_data()
