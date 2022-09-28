from random import random
import torch
import pickle
from transformers import BertTokenizer
import json

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

def check_pkl_length(data_dir, label):
    f = open(data_dir, 'rb')
    features = pickle.load(f)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long).tolist()
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long).tolist()
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long).tolist()
    answerStart = torch.tensor([f[3] for f in features], dtype=torch.long).tolist()
    answerEnd = torch.tensor([f[4] for f in features], dtype=torch.long).tolist()
    number, length, start, end = 0, 0.0, 0.0, 0.0
    for i in range(len(features)):
        context_ids = all_input_ids[i]
        context = tokenizer.convert_ids_to_tokens(context_ids)
        j = 0
        for j in range(len(context)):
            if context[j] == '[PAD]':
                break
        number += 1
        length += j
        start += answerStart[i]
        end += answerEnd[i]
    print(label+' 数量:', number)
    print(label+' pkl平均编码长度:', length / number)
    print(label+' 答案平均开始位置:', start / number)
    print(label+' 答案平均结束位置:', end / number)

    # random的答案大多数在context末尾，答案并没有随机分布在512之间，这个问题有两种解决方法
    # 1：重新生成reasonable和random line的数据，而不是sentence，之前写错了，推荐这种方法
    # 2：重新生成random的数据，使其不靠后分布，这个做得快一点，先做这个
    #    这个实现了不咋行，接下来探索第一个想法

if __name__ == '__main__':
    data_dirs = ['./data/newQuestionGeneration/bridge/pretuning_data_train_new4.pkl', './data/dataFeatures/bridge/constructDomainTrainData_new5.pkl']
    data_labels = ['reasonable', 'random']
    for i, data_dir in enumerate(data_dirs):
        check_pkl_length(data_dir, data_labels[i])

