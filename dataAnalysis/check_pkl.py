import pickle

# seeds = [43, 83, 271, 659, 859]
# for seed in seeds:
#     f = open('./data/XSJModel/bridge/train_'+str(seed)+'.pkl', 'rb')
#     features = pickle.load(f)
#     # train
#     length = len(features)
#     all_input_ids = [f[0] for f in features]
#     all_input_mask = [f[1] for f in features]
#     all_label_ids = [f[2] for f in features]
#     for i in range(len(all_input_ids)):
#         if len(all_input_ids[i]) != 512:
#             print(i, 'error')
#         if len(all_input_mask[i]) != 512:
#             print(i, 'error')
#         if len(all_label_ids[i]) != 512:
#             print(i, 'error')
f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_train_data.pkl', 'rb')
features = pickle.load(f)
# train
length = len(features)
f.close()
print(length)
# all_input_ids = [f[0] for f in features]
# all_input_mask = [f[1] for f in features]
# all_label_ids = [f[2] for f in features]
# for i in range(len(all_input_ids)):
#     if len(all_input_ids[i]) != 512:
#         print(i, 'error')
#     if len(all_input_mask[i]) != 512:
#         print(i, 'error')
#     if len(all_label_ids[i]) != 512:
#         print(i, 'error')
#     print(len(all_input_ids[i]))
#     break