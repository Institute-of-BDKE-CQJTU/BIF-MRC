import torch
from torch import nn
from transformers import BertTokenizer, AdamW,BertForQuestionAnswering
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
        self.model = BertForQuestionAnswering.from_pretrained(self.modelPath)
        self.outputPath = arg.model_output_path
        self.cuda = int(arg.cuda)
        device = torch.device('cuda:'+str(arg.cuda)) if torch.cuda.is_available() else torch.device('cpu')
        for name ,param in self.model.named_parameters():
            if str(name)[:4] == 'bert':
                param.requires_grad = False
        self.model.to(device)
        # self.optim = AdamW(self.model.parameters(), lr=self.lr)
        self.optim = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            loop = tqdm(enumerate(train_loader), total =len(train_loader))
            for iter, data in loop:
                loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
                data = [d.cuda(self.cuda) for d in data]
                self.optim.zero_grad()
                input_ids, attention_mask, token_type_ids, answerStart, answerEnd = data
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=answerStart, end_positions=answerEnd)
                loss = outputs.loss
                # if iter%5 == 0 and iter!= 0:
                #     print('epoch:{0},loss:{1}'.format(epoch, loss))
                loop.set_postfix(loss = loss.item())
                loss.backward()
                self.optim.step()

    def save(self):
        output_model_file = self.outputPath+'/'+self.trainData +"/"+str(self.cuda)+ "/modelForIdea.bin"
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), output_model_file)

        
def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/finalCleanQuestionGeneration/bridge/answer_predict_test_512.pkl', 'rb')
    features = pickle.load(f)
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    file=open('./finalCleanQuestionGeneration/predict_start_index_512.txt','r')
    predict_start_index_str = file.readlines()[0].strip()[1:-1]
    predict_start_index = list(map(int, predict_start_index_str.split(',')))
    file.close()

    file=open('./finalCleanQuestionGeneration/predict_end_index_512.txt','r')
    predict_end_index_str = file.readlines()[0].strip()[1:-1]
    predict_end_index = list(map(int, predict_end_index_str.split(',')))
    file.close()
    answerStart = torch.tensor(predict_start_index, dtype=torch.long)
    answerEnd = torch.tensor(predict_end_index, dtype=torch.long)
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, answerStart, answerEnd)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    f.close()
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    trainer.save()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--model_input_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./idea/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=1)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--part', type=float, default=1)
    parse.add_argument('--cuda', type=int, default=0)

    arg = parse.parse_args()
    main(arg)
