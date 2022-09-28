# from transformers import BertTokenizer, BartForQuestionAnswering
# import torch 

# tokenizer = BertTokenizer.from_pretrained('./model/bart-base-chinese')
# model = BartForQuestionAnswering.from_pretrained('./model/bart-base-chinese')

# question, text = "桥长多少米?", "A桥长1200m"
# inputs = tokenizer(question, text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')

# start_positions = torch.tensor([1])
# end_positions = torch.tensor([3])

# outputs = model(inputs['input_ids'], inputs['attention_mask'], start_positions=start_positions, end_positions=end_positions)
# loss = outputs.loss
# start_scores = outputs.start_logits
# end_scores = outputs.end_logits

import torch
from torch import nn
from transformers import BertTokenizer, AdamW,BartForQuestionAnswering
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
        self.model = BartForQuestionAnswering.from_pretrained(self.modelPath)
        self.outputPath = arg.model_output_path
        self.cuda = int(arg.cuda)
        device = torch.device('cuda:'+str(arg.cuda)) if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.optim = AdamW(self.model.parameters(), lr=self.lr)

    def train(self, train_loader):
        self.model.train()
        for epoch in range(self.epochs):
            loop = tqdm(enumerate(train_loader), total =len(train_loader))
            for iter, data in loop:
                loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
                data = [d.cuda(self.cuda) for d in data]
                self.optim.zero_grad()
                input_ids, attention_mask, answerStart, answerEnd = data
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, start_positions=answerStart, end_positions=answerEnd)
                loss = outputs.loss
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
    if arg.postfix_number == 4:
        f = open('./data/finalCleanQuestionGeneration/bridge/pretuning_train_data_4_3808.pkl', 'rb')
    elif arg.postfix_number == 5:
        f = open('./data/finalCleanQuestionGeneration/bridge/pretuning_train_data_5_3808.pkl', 'rb')
    elif arg.postfix_number == 6:
        f = open('./data/finalCleanQuestionGeneration/bridge/pretuning_train_data_6_3808.pkl', 'rb')
    else:
        print('指定postfix_number错误')
        return
    features = pickle.load(f)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    answerStart = torch.tensor([f[3] for f in features], dtype=torch.long)
    answerEnd = torch.tensor([f[4] for f in features], dtype=torch.long)
    trainDataset = TensorDataset(all_input_ids, all_input_mask, answerStart, answerEnd)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size)
    f.close()
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    trainer.save()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--model_input_path', type=str, default='./model/bart-base-chinese', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./bart/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=1)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--part', type=float, default=1)
    parse.add_argument('--cuda', type=int, default=1)
    parse.add_argument('--postfix_number', type=int, default=4)

    arg = parse.parse_args()
    main(arg)
