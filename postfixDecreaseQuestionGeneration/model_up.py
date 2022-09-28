from typing import List
import torch
from torch import nn
import sys
sys.path.append('./')
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from finetune.evaluate import *
import argparse
import pickle
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_path
        self.cuda = int(arg.cuda)
        self.modelForQA = BertForQuestionAnswering.from_pretrained(arg.model_path)
        self.modelForQA.load_state_dict(torch.load('./idea/model/bridge/'+str(self.cuda)+'/modelForIdea.bin'))
        device = torch.device('cuda:'+str(arg.cuda)) if torch.cuda.is_available() else torch.device('cpu')
        self.optim = AdamW(self.modelForQA.parameters(), lr=self.lr)
        self.modelForQA.to(device)
        self.final_result = dict()

    def train(self, train_loader):
        self.modelForQA.train()
        for epoch in range(self.epochs):
            loop = tqdm(enumerate(train_loader), total = len(train_loader))
            for iter, data in loop:
                loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
                data = [d.cuda(self.cuda) for d in data]
                self.optim.zero_grad()
                input_ids, attention_mask, token_type_ids, answerStart, answerEnd = data
                outputs = self.modelForQA(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=answerStart, end_positions=answerEnd)
                loss = outputs.loss
                loop.set_postfix(loss_value = loss.item())
                loss.backward()
                self.optim.step()

    def test(self, test_loader):
        self.modelForQA.eval()
        starts, ends = [], []
        with torch.no_grad():
            for iter, data in enumerate(test_loader):
                input_ids, attention_mask, token_type_ids = data
                input_ids, attention_mask, token_type_ids = input_ids.cuda(self.cuda), attention_mask.cuda(self.cuda), token_type_ids.cuda(self.cuda)
                outputs = self.modelForQA(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                start = outputs.start_logits.cpu().argmax(dim=1)
                end = outputs.end_logits.cpu().argmax(dim=1)
                starts.extend(start.tolist())
                ends.extend(end.tolist())
        return starts, ends

    def set_final_result(self, question_ids:List[str], inputs_tokens:List[List[str]], starts:List[int], ends:List[int], answers:List[str]):
        for i, question_id in enumerate(question_ids):
            self.final_result[question_id] = dict()
            predict_answer_tokens = inputs_tokens[i][starts[i]:ends[i]+1]
            predict_answer = ''.join(tokenizer.convert_tokens_to_string(predict_answer_tokens)).replace(' ', '')
            EM = calc_em_score(answers[i], predict_answer)
            F1 = calc_f1_score(answers[i], predict_answer)
            self.final_result[question_id]['EM'] = EM
            self.final_result[question_id]['F1'] = F1

    def get_final_result(self, question_ids:List[str]):
        count = 0
        total_EM, total_F1 = 0.0, 0.0
        i = 0
        while i < len(question_ids):
            EM, F1 = self.final_result[question_ids[i]]['EM'], self.final_result[question_ids[i]]['F1'] 
            if i+1 < len(question_ids) and question_ids[i+1].split('-')[0] == question_ids[i].split('-')[0]:
                j = 0
                for j in range(i+1, len(question_ids)):
                    if question_ids[j].split('-')[0] != question_ids[i].split('-')[0]:
                        break
                    if self.final_result[question_ids[j]]['F1'] > F1:
                        EM, F1 = self.final_result[question_ids[j]]['EM'], self.final_result[question_ids[j]]['F1']
                # 如果是最后都取完了，直接退出
                if j == len(question_ids) - 1 and question_ids[j].split('-')[0] == question_ids[i].split('-')[0]:
                    i = j
                else:
                    i = j-1
            i += 1
            count += 1
            total_EM += EM
            total_F1 += F1
        return total_F1/ count, total_EM / count
    
    def test_get_final_result(self, question_ids, temp_json):
        self.final_result = temp_json
        return self.get_final_result(question_ids)
        

def test_json():
    temp_json = {
        '0-0':{
            'EM':0, 'F1':0.5
        },
        '0-1':{
            'EM':1, 'F1':1
        },
        '1-0':{
            'EM':0, 'F1':0.65
        },
        '2-0':{
            'EM':0, 'F1':0.7
        },
        '2-1':{
            'EM':0, 'F1':0.5
        },
        '2-2':{
            'EM':0, 'F1':0.8
        },
        '3-0':{
            'EM':1, 'F1':1
        }
    }
    question_ids = ['0-0', '0-1', '1-0', '2-0', '2-1', '2-2', '3-0']
    trainer = trainModel(arg)
    F1, EM = trainer.test_get_final_result(question_ids, temp_json)
    print('F1:', F1, '\n', 'EM:', EM)




def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/XSJModel/bridge/train_'+str(arg.seed)+'.pkl', 'rb')
    features = pickle.load(f)
    # train
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[1] for f in features], dtype=torch.long)
    answerStart = torch.tensor([f[3] for f in features], dtype=torch.long)
    answerEnd = torch.tensor([f[4] for f in features], dtype=torch.long)
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, answerStart, answerEnd)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    f.close()
    
    # test
    f = open('./data/XSJModel/bridge/test_'+str(arg.seed)+'.pkl', 'rb')
    features = pickle.load(f)
    # eval
    question_ids = [f[0] for f in features]
    input_tokens = [f[1] for f in features]
    all_input_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[4] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[3] for f in features], dtype=torch.long)
    answers = [f[5] for f in features]
    f.close()
    testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    starts, ends = trainer.test(testDataLoader)
    trainer.set_final_result(question_ids, input_tokens, starts, ends, answers)
    EM, F1 = trainer.get_final_result(question_ids)
    print('F1:{:.4f}\nEM:{:.4f}'.format(F1, EM))
    

if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--model_input_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    # parse.add_argument('--model_path', type=str, default='./model/chinese_roberta_wwm_ext', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./idea/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=2)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=43)
    parse.add_argument('--cuda', type=int, default=1)

    arg = parse.parse_args()
    main(arg)
    # test_json()