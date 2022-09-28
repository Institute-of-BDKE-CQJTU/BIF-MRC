import json
import pickle



# datas = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']
datas = json.load(open('./data/newQuestionGeneration/bridge/question_predict_train.json', 'r', encoding='UTF-8'))['data']
number, length = 0, 0.0
count = len(datas)
postfix_set = set()
numbers = [0]*6
for i in range(count):
    numbers[datas[i]['postfix_index']] += 1
print(numbers)
# datas = json.load(open('./data/newQuestionGeneration/bridge/pretuning_data_train.json', 'r', encoding='UTF-8'))['data']
# f = open('./data/dataFeatures/bridge/constructDomainTrainData.pkl', 'rb')
# f = open('./data/newQuestionGeneration/bridge/pretuning_data_train.pkl', 'rb')
# features = pickle.load(f)
# f.close()
# 真实数据问题平均长度为12.6
# 生成的数据问题平均长度为56.6
# 两个数据长度差远了