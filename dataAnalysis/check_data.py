import json
import pickle

# f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_train_data_6.pkl', 'rb')
# # f = open('./data/finalCleanQuestionGeneration/bridge/answer_predict_train_data_6.pkl', 'rb')
# features = pickle.load(f)
# f.close()
# print(len(features))

datas = json.load(open('./data/finalCleanQuestionGeneration/bridge/question_predict_train.json', 'r', encoding='UTF-8'))['data']
count = len(datas)
print(count)