import pickle

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
