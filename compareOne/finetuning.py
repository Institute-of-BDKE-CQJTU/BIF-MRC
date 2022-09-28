from model import ModelWithQASSHead
from transformers import BertTokenizer
import torch
from torch import nn
from transformers import AdamW
import argparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pickle
from tqdm import tqdm
import sys
sys.path.append('./')
from finetune.evaluate import *

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.model = ModelWithQASSHead.from_pretrained(self.modelPath)
        self.cuda = int(arg.cuda)
        self.model.load_state_dict(torch.load('./compareOne/model/bridge/'+str(self.cuda)+'/modelForIdea.bin'))
        self.outputPath = arg.model_output_path
        device = torch.device('cuda:'+str(arg.cuda)) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.optim = AdamW(self.model.parameters(), lr=self.lr)

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            for iter, data in enumerate(train_loader):
                data = [d.cuda(self.cuda) for d in data]
                self.optim.zero_grad()
                input_ids, attention_mask, masked_position, answerStart, answerEnd = data
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, masked_positions=masked_position, start_positions=answerStart, end_positions=answerEnd)
                loss = outputs['loss']
                loss.backward()
                self.optim.step()

    def test(self, test_loader):
        self.model.eval()
        start_index, end_index = [], []
        with torch.no_grad():
            for iter, data in enumerate(test_loader):
                data = [d.cuda(self.cuda) for d in data]
                input_ids, attention_mask, masked_position = data
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, masked_positions=masked_position)
                start = outputs['start_logits'].cpu().argmax(dim=1).tolist()
                end = outputs['end_logits'].cpu().argmax(dim=1).tolist()
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
        print('F1:{:.4f},EM:{:.4f}'.format(F1/count, EM/count))

        
def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    # 训练
    if arg.full_data == True:
        f = open('./data/compareOne/bridge/train.pkl', 'rb')
    else:
        f = open('./data/compareOne/bridge/train-'+str(arg.train_data_size)+'-'+str(arg.seed)+'.pkl', 'rb')
    features = pickle.load(f)
    f.close()
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    masked_positions = torch.tensor([f[2] for f in features], dtype=torch.long)
    answerStart = torch.tensor([f[3] for f in features], dtype=torch.long)
    answerEnd = torch.tensor([f[4] for f in features], dtype=torch.long)
    trainDataset = TensorDataset(all_input_ids, all_input_mask, masked_positions, answerStart, answerEnd)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    
    # 测试
    f = open('./data/compareOne/bridge/dev.pkl', 'rb')
    features = pickle.load(f)
    f.close()
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    masked_positions = torch.tensor([f[2] for f in features], dtype=torch.long)
    input_tokens = [f[3] for f in features]
    answers = [f[4] for f in features]
    testDataset = TensorDataset(all_input_ids, all_input_mask, masked_positions)
    testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    predict_start_index, predict_end_index = trainer.test(testDataLoader)
    trainer.eval(predict_start_index, predict_end_index, input_tokens, answers)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--model_input_path', type=str, default='./model/chinese_macbert_base', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./compareOne/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--train_data_size', type=int, default=16)
    parse.add_argument('--epochs', type=int, default=10)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--cuda', type=int, default=1)
    parse.add_argument('--full_data', type=bool, default=False)
    parse.add_argument('--seed', type=int, default=43)

    arg = parse.parse_args()
    main(arg)

