import json

questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
postfix_to_index = dict(zip(questionPostfix, list(range(6))))

file=open('./test/pretuning_index2.txt','r')
pretuning_index_str = file.readlines()[0].strip()[1:-1]
pretuning_index = list(map(int, pretuning_index_str.split(',')))
file.close()

file=open('./test/origin_index2.txt','r')
origin_index_str = file.readlines()[0].strip()[1:-1]
origin_index = list(map(int, origin_index_str.split(',')))
file.close()

file=open('./test/pretuning_question_length.txt','r')
pretuning_question_length_str = file.readlines()[0].strip()[1:-1]
pretuning_question_length = list(map(int, pretuning_question_length_str.split(',')))
file.close()


def check_data(data_dir, label):
    datas = json.load(open(data_dir, 'r', encoding='UTF-8'))['data']
    print(label+'第一条数据:\n', datas[0])
    # reasonable第一条数据:
    # {'context': '编号原则为了便于检测记录和描述，桥梁结构构件编号规则如下：1、路线方向是结构构件编号规则的基础，沿养护管理规定的路线方向区分左侧和右侧', 'postfix': '维修建议？', 'answer': '左侧和右侧', 'answer_start': 61}
    # random第一条数据:
    # {'context': '编号原则为了便于检测记录和描述，桥梁结构构件编号规则如下：1、路线方向是结构构件编号规则的基础，沿养护管理规定的路线方向区分左侧和右侧', 'question': '编号原则为了便于检测记录和描述，桥梁结构构件编号规则如下：1、路线方向是结构构件编号规则的基础，沿养护管理规定的路线方向是多少？', 'answer': '区分左侧和右侧'}

def check_long_and_short_answer_length(data_dir, label):
    datas = json.load(open(data_dir, 'r', encoding='UTF-8'))['data']
    count = len(datas)
    short_answer_length, long_answer_length = 0.0, 0.0
    long_answer_number, short_answer_number = 0, 0
    if label == 'reasonable':
        for i in range(count):
            if i not in pretuning_index:
                continue
            if postfix_to_index[datas[i]['postfix']] > 2:
                long_answer_length += len(datas[i]['answer'])
                long_answer_number += 1
            else:
                short_answer_length += len(datas[i]['answer'])
                short_answer_number += 1
    else:
        for i in range(count):
            if i not in origin_index:
                continue
            j = 0
            for j in range(6):
                if questionPostfix[j] in datas[i]['question']:
                    break
            if j > 2:
                long_answer_length += len(datas[i]['answer'])
                long_answer_number += 1
            else:
                short_answer_length += len(datas[i]['answer'])
                short_answer_number += 1
    print(label+'长答案个数:', long_answer_number)
    print(label+'长答案平均长度:', long_answer_length/long_answer_number)
    print(label+'短答案个数:', short_answer_number)
    print(label+'短答案平均长度:', short_answer_length/short_answer_number)
    # reasonable长答案个数: 46446
    # reasonable长答案平均长度: 22.428153124058046
    # reasonable短答案个数: 34035
    # reasonable短答案平均长度: 11.32792713383282

    # random长答案个数: 40292
    # random长答案平均长度: 18.977340414970715
    # random短答案个数: 40189
    # random短答案平均长度: 10.048744681380477
    # 长短不均衡

def check_question_type_number(data_dir, label):
    numbers = [0, 0, 0, 0, 0, 0]
    datas = json.load(open(data_dir, 'r', encoding='UTF-8'))['data']
    count = len(datas)
    if label == 'reasonable':
        for i in range(count):
            numbers[postfix_to_index[datas[i]['postfix']]] += 1
    else:
        for i in range(count):
            j = 0
            for j in range(6):
                if questionPostfix[j] in datas[i]['question']:
                    break
            numbers[j] += 1
    print(label+'各类型问题答案个数:', numbers)
    # reasonable各类型问题答案个数: [28229, 1284, 4522, 6594, 24976, 14876]
    # random各类型问题答案个数: [13421, 13326, 13442, 13336, 13608, 13348]

def check_question_length(data_dir, label):
    datas = json.load(open(data_dir, 'r', encoding='UTF-8'))['data']
    count = len(datas)
    length, number = 0.0, 0
    if label == 'reasonable':
        for i in range(len(pretuning_question_length)):
            length += pretuning_question_length[i]
            number += 1
    else:
        for i in range(count):
            if i not in origin_index:
                continue
            length += len(datas[i]['question'])
            number += 1
    print(label+'问题长度:', length/number) 
    
    

if __name__ == "__main__":
    data_dirs = ['./data/newQuestionGeneration/bridge/pretuning_data_train.json', './data/preprocessingData/bridge/constructDomainTrainData.json']
    data_labels = ['reasonable', 'random']
    for i, data_dir in enumerate(data_dirs):
        # check_data(data_dir, data_labels[i])
        check_long_and_short_answer_length(data_dir, data_labels[i])
        # check_question_type_number(data_dir, data_labels[i])
        check_question_length(data_dir, data_labels[i])