from transformers import AlbertForQuestionAnswering, BertForQuestionAnswering, AlbertForSequenceClassification, AdamW, BertForSequenceClassification, BertTokenizer
import torch
import argparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pickle
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.cuda = int(arg.cuda)
        # self.model = AlbertForSequenceClassification.from_pretrained(self.modelPath, num_labels=7)
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

    def eval(self, predict_start_index, predict_end_index, true_start_index, true_end_index):
        length = len(predict_end_index)
        em = 0.0
        for i in range(length):
            if predict_start_index[i] == true_start_index[i] and predict_end_index[i] == true_end_index[i]:
                em += 1.0
        print('em:', em / length)

    def save(self):
        output_model_file = self.outputPath+'/'+self.trainData +'/'+str(self.cuda)+ "/modelForIdea.bin"
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), output_model_file)


def main_test(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/newQuestionGeneration/bridge/answer_predict_train.pkl', 'rb')
    features = pickle.load(f)
    f.close()
    print(features[0])
    # train
    # length = len(features)
    # all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    # all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    # all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    # answerStart = torch.tensor([f[3] for f in features], dtype=torch.long)[:int(length*arg.part)]
    # answerEnd = torch.tensor([f[4] for f in features], dtype=torch.long)[:int(length*arg.part)]
    # trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, answerStart, answerEnd)
    # trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    
    # trainer = trainModel(arg)
    # trainer.train(trainDataLoader)
    # f.close()
    # # eval
    # all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    # all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    # all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    # true_answer_start = [f[3] for f in features][int(length*arg.part):]
    # true_answer_end = [f[4] for f in features][int(length*arg.part):]
    # testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    # testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    # predict_start_index, predict_end_index = trainer.test(testDataLoader)
    # trainer.eval(predict_start_index, predict_end_index, true_answer_start, true_answer_end)


def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/newQuestionGeneration/bridge/answer_predict_train.pkl', 'rb')
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
    f = open('./data/newQuestionGeneration/bridge/answer_predict_test.pkl', 'rb')
    features = pickle.load(f)
    # eval
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    predict_start_index, predict_end_index = trainer.test(testDataLoader)
    f.close()
    file=open('./newQuestionGeneration/predict_start_index.txt','w')  
    file.write(str(predict_start_index))
    file.close()

    file=open('./newQuestionGeneration/predict_end_index.txt','w')  
    file.write(str(predict_end_index))
    file.close()
    trainer.save()
    

def tokenizer_test_data():
    tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')
    datas = json.load(open('./data/preprocessingData/bridge/constructDomainTrainData4.json', 'r', encoding='UTF-8'))['data']
    count = len(datas)
    contexts, features = [], []
    questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
    file=open('./newQuestionGeneration/question_postfix.txt','r')
    question_postfix_predict_str = file.readlines()[0].strip()[1:-1]
    question_postfix_predict = list(map(int, question_postfix_predict_str.split(',')))
    postfixs = [questionPostfix[i] for i in question_postfix_predict]
    file.close()
    for i in range(count):
        contexts.append(datas[i]['context'])
    tokenizer_data = tokenizer(postfixs, contexts, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    input_ids, attention_mask, token_type_ids = tokenizer_data['input_ids'].tolist(), tokenizer_data['attention_mask'].tolist(), tokenizer_data['token_type_ids'].tolist()
    for i in range(len(contexts)):
        features.append([input_ids[i], attention_mask[i], token_type_ids[i]])
    with open('./data/newQuestionGeneration/bridge/answer_predict_test.pkl', 'wb') as f:
        pickle.dump(features, f)

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
    parse.add_argument('--cuda', type=int, default=0)
    parse.add_argument('--part', type=float, default=0.8)

    arg = parse.parse_args()
    # tokenizer_test_data()
    # main(arg)
    main_test(arg)
