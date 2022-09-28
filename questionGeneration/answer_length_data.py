# 这个文件tokenizer数据，以供长度预测

import json
import pickle
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']

# 先将训练数据准备好
def prepare_train_data():
    datas = json.load(open('./data/questionData/bridge/train.json', 'r', encoding='UTF-8'))['data']
    question, context, length = [], [], []
    for data in datas:
        question.append(data['question'])
        context.append(data['context'])
        length.append(data['answerLength'])

    input = tokenizer(question, context,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    features = []
    for i in range(len(question)):
        features.append([input['input_ids'][i].tolist(), input['attention_mask'][i].tolist(), input['token_type_ids'][i].tolist(), length[i]])

    with open('./data/questionData/bridge/answer_length_train.pkl', 'wb') as f:
        pickle.dump(features, f)

# 准备测试数据
def prepare_test_data():
    datas = json.load(open('./data/preprocessingData/bridge/constructDomainTrainDataNoPostfix.json', 'r', encoding='UTF-8'))['data']
    print(datas[3])
    # file=open('./questionGeneration/question_postfix.txt','r')
    # question_postfix_predict_str = file.readlines()[0].strip()[1:-1]
    # question_postfix_predict = list(map(int, question_postfix_predict_str.split(',')))
    # file.close()

if __name__ == "__main__":
    prepare_test_data()