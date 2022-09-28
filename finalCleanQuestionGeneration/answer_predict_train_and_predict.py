from transformers import BertForQuestionAnswering, AdamW, BertTokenizer
import torch
import argparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pickle
import json
from tqdm import tqdm
import sys
sys.path.append('./')
from finetune.evaluate import *
from sklearn.metrics import accuracy_score, recall_score, f1_score

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.cuda = int(arg.cuda)
        self.model = BertForQuestionAnswering.from_pretrained(self.modelPath)
        self.outputPath = arg.model_output_path
        device = torch.device('cuda:'+str(arg.cuda)) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.optim = AdamW(self.model.parameters(), lr=self.lr)

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            loop = tqdm(enumerate(train_loader), total = len(train_loader))
            for iter, data in loop:
                loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
                data = [d.cuda(self.cuda) for d in data]
                self.optim.zero_grad()
                input_ids, attention_mask, token_type_ids, answerStart, answerEnd  = data
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=answerStart, end_positions=answerEnd)
                loss = outputs.loss
                loop.set_postfix(loss_value = loss.item())
                loss.backward()
                self.optim.step()

    def test(self, test_loader):
        self.model.eval()
        start_index, end_index = [], []
        with torch.no_grad():
            for iter, data in enumerate(test_loader):
                input_ids, attention_mask, token_type_ids = data
                input_ids, attention_mask, token_type_ids = input_ids.cuda(self.cuda), attention_mask.cuda(self.cuda), token_type_ids.cuda(self.cuda)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                start = outputs.start_logits.cpu().argmax(dim=1).tolist()
                end = outputs.end_logits.cpu().argmax(dim=1).tolist()
                start_index.extend(start)
                end_index.extend(end)
        return start_index, end_index

    def eval(self, predict_start_index, predict_end_index, input_tokens, answers):
        EM, F1 = 0.0, 0.0
        count = 0
        for i in range(len(answers)):
            predict_answer = tokenizer.convert_tokens_to_string(input_tokens[i][predict_start_index[i]:predict_end_index[i]+1])
            predict_answer = predict_answer.replace(' ', '')
            EM += calc_em_score([answers[i]], predict_answer)
            F1 += calc_f1_score([answers[i]], predict_answer)
            count += 1
        print('EM:', EM/count)
        print('F1:', F1/count)

def main_test(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/finalCleanQuestionGeneration/bridge/answer_predict_train_data_6.pkl', 'rb')
    features = pickle.load(f)
    f.close()
    # train
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    answerStart = torch.tensor([f[3] for f in features], dtype=torch.long)[:int(length*arg.part)]
    answerEnd = torch.tensor([f[4] for f in features], dtype=torch.long)[:int(length*arg.part)]
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, answerStart, answerEnd)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    # test
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    true_answer_start = [f[3] for f in features][int(length*arg.part):]
    true_answer_end = [f[4] for f in features][int(length*arg.part):]
    input_tokens = [f[5] for f in features][int(length*arg.part):]
    answers = [f[6] for f in features][int(length*arg.part):]
    testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    predict_start_index, predict_end_index = trainer.test(testDataLoader)
    trainer.eval(predict_start_index, predict_end_index, input_tokens, answers)

def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/finalCleanQuestionGeneration/bridge/answer_predict_train_data_6.pkl', 'rb')
    features = pickle.load(f)
    # train
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    answerStart = torch.tensor([f[3] for f in features], dtype=torch.long)
    answerEnd = torch.tensor([f[4] for f in features], dtype=torch.long)
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, answerStart, answerEnd)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size)
    
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    f.close()
    f = open('./data/finalCleanQuestionGeneration/bridge/question_type_random_for_answer_predict_test_256_6.pkl', 'rb')
    features = pickle.load(f)
    # eval
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    predict_start_index, predict_end_index = trainer.test(testDataLoader)
    f.close()
    file=open('./finalCleanQuestionGeneration/predict_start_index_256_6.txt','w')
    file.write(str(predict_start_index))
    file.close()

    file=open('./finalCleanQuestionGeneration/predict_end_index_256_6.txt','w')
    file.write(str(predict_end_index))
    file.close()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    # parse.add_argument('--model_input_path', type=str, default='./model/albert_chinese_small', help='模型选择')
    parse.add_argument('--model_input_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./newQuestionGeneration/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=1)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--part', type=float, default=0.8)
    parse.add_argument('--cuda', type=int, default=0)

    arg = parse.parse_args()
    main(arg)
    # main_test(arg)
