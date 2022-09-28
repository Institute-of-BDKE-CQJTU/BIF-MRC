from transformers import AlbertForQuestionAnswering, AlbertForSequenceClassification, AdamW, BertForSequenceClassification, BertTokenizer
import torch
import argparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
import json

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.cuda = int(arg.cuda)
        # self.model = AlbertForSequenceClassification.from_pretrained(self.modelPath, num_labels=6)
        self.model = BertForSequenceClassification.from_pretrained(self.modelPath, num_labels=7)
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
                input_ids, attention_mask, token_type_ids, labels = data
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
                loss = outputs.loss
                loop.set_postfix(loss_value = loss.item())
                loss.backward()
                self.optim.step()

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

    def eval(self, predict_type, eval_type):
        print('accuracy:', accuracy_score(eval_type, predict_type))
        print('recall:', recall_score(eval_type, predict_type, average='micro'))
        print('f1 score:', f1_score(eval_type, predict_type, average='micro'))

        
def main_test(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_train_data_6.pkl', 'rb')
    features = pickle.load(f)
    # train
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    labels = torch.tensor([f[3] for f in features], dtype=torch.long)[:int(length*arg.part)]
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, labels)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    f.close()
    # test
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    labels = torch.tensor([f[3] for f in features], dtype=torch.long)[int(length*arg.part):].tolist()
    testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    predict_type = trainer.test(testDataLoader)
    trainer.eval(predict_type, labels)

def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_train_data_6.pkl', 'rb')
    features = pickle.load(f)
    # train
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    labels = torch.tensor([f[3] for f in features], dtype=torch.long)
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, labels)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    f.close()

    f = open('./data/finalCleanQuestionGeneration/bridge/question_type_predict_test_data.pkl', 'rb')
    features = pickle.load(f)
    # test
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    predict_type = trainer.test(testDataLoader)
    f.close()
    file=open('./finalCleanQuestionGeneration/question_postfix_6.txt','w')  
    file.write(str(predict_type))
    file.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    # parse.add_argument('--model_input_path', type=str, default='./model/albert_chinese_small', help='模型选择')
    parse.add_argument('--model_input_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./finalCleanQuestionGeneration/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=1)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--cuda', type=int, default=0)
    parse.add_argument('--part', type=float, default=0.8)

    arg = parse.parse_args()
    # tokenizer_test_data()
    main(arg)
    # main_test(arg)