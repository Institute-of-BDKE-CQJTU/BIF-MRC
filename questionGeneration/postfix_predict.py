# 预测合理的后缀

import json
import torch
from torch import nn
from transformers import BertTokenizer, AdamW,BertForSequenceClassification
import argparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pickle
from tqdm import tqdm

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.cuda = int(arg.cuda)
        self.model = BertForSequenceClassification.from_pretrained(self.modelPath, num_labels=6)
        self.model.load_state_dict(torch.load('./questionGeneration/model/bridge/'+str(self.cuda)+'/modelForIdea.bin'))
        self.outputPath = arg.model_output_path
        device = torch.device('cuda:'+str(arg.cuda)) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.optim = AdamW(self.model.parameters(), lr=self.lr)
        # self.optim = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    def test(self, test_loader):
        self.model.eval()
        predict_type = []
        with torch.no_grad():
            for iter, data in enumerate(test_loader):
                input_ids, attention_mask, token_type_ids = data
                input_ids, attention_mask, token_type_ids = input_ids.cuda(self.cuda), attention_mask.cuda(self.cuda), token_type_ids.cuda(self.cuda)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                classification_type = outputs.logits.cpu().argmax(1).tolist()
                predict_type.extend(classification_type)
        return predict_type
            

def wipe_out_postfix(question):
    questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
    for i in range(6):
        if questionPostfix[i] in question:
            return question[:-len(questionPostfix[i])]

def get_test_data():
    datas = json.load(open('./data/preprocessingData/bridge/constructDomainTrainData.json', 'r', encoding='UTF-8'))['data']
    count = len(datas)
    prefixs = []
    contexts = []
    for i in range(count):
        prefixs.append(wipe_out_postfix(datas[i]['question']))
        contexts.append(datas[i]['context'])
    return prefixs, contexts

def tokenizer_data(prefixs, contexts):
    tokenizer = BertTokenizer.from_pretrained("./model/chinese_bert_wwm")
    tokenizer_data = tokenizer(prefixs, contexts,padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids, attention_mask, token_type_ids = tokenizer_data['input_ids'].tolist(), tokenizer_data['attention_mask'].tolist(), tokenizer_data['token_type_ids'].tolist()
    features = []
    count = len(prefixs)
    for i in range(count):
        features.append([input_ids[i], attention_mask[i], token_type_ids[i]])
    with open('./data/dataFeatures/bridge/get_question_postfix.pkl', 'wb') as f:
        pickle.dump(features, f)


        
def main(arg):
    # prefixs, contexts = get_test_data()
    # tokenizer_data(prefixs, contexts)
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/dataFeatures/'+arg.train_data+'/get_question_postfix.pkl', 'rb')
    features = pickle.load(f)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size)
    f.close()
    trainer = trainModel(arg)
    question_types = trainer.test(trainDataLoader)
    file=open('./questionGeneration/question_postfix.txt','w')  
    file.write(str(question_types))
    file.close()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--model_input_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./questionGeneration/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=1)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--cuda', type=int, default=1)

    arg = parse.parse_args()
    main(arg)