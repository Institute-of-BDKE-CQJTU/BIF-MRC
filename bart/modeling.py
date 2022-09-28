import torch
from torch import nn
from transformers import BertTokenizer, BartForQuestionAnswering, AdamW
from evaluate import *

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_path
        self.cuda = int(arg.cuda)
        self.modelForQA = BartForQuestionAnswering.from_pretrained(arg.model_path)
        if arg.pre == True:
            self.modelForQA.load_state_dict(torch.load('./bart/model/bridge/'+str(self.cuda)+'/modelForIdea.bin'))
        device = torch.device('cuda:'+str(arg.cuda)) if torch.cuda.is_available() else torch.device('cpu')
        if arg.lock == True:
            for name, param in self.modelForQA.named_parameters():
                if str(name)[:5] == 'model':
                    param.requires_grad = False
            self.optim = AdamW(filter(lambda p: p.requires_grad, self.modelForQA.parameters()), lr=self.lr)
        else:
            self.optim = AdamW(self.modelForQA.parameters(), lr=self.lr)
        self.modelForQA.to(device)

    def train(self, train_loader):
        self.modelForQA.train()
        for epoch in range(self.epochs):
            for iter, data in enumerate(train_loader):
                data = [d.cuda(self.cuda) for d in data]
                self.optim.zero_grad()
                input_ids, attention_mask, token_type_ids, answerStart, answerEnd = data
                outputs = self.modelForQA(input_ids=input_ids, attention_mask=attention_mask, start_positions=answerStart, end_positions=answerEnd)
                loss = outputs.loss
                if iter%5 == 0 and iter!= 0:
                    print('epoch:{0},loss:{1}'.format(epoch, loss))
                loss.backward()
                self.optim.step()

    def test(self, test_loader):
        self.modelForQA.eval()
        tokenizer = BertTokenizer.from_pretrained(self.modelPath)
        count = 0
        EM = 0.0
        F1 = 0.0
        with torch.no_grad():
            for iter, data in enumerate(test_loader):
                input_ids, attention_mask, token_type_ids, answers = data
                cpuInput_ids = input_ids
                input_ids, attention_mask = input_ids.cuda(self.cuda), attention_mask.cuda(self.cuda)
                outputs = self.modelForQA(input_ids=input_ids, attention_mask=attention_mask)
                start = outputs.start_logits.cpu().argmax(dim=1)
                end = outputs.end_logits.cpu().argmax(dim=1)
                # 求开始位置和终止位置
                count += len(cpuInput_ids)
                for i in range(len(cpuInput_ids)):
                    tokens = tokenizer.convert_ids_to_tokens(cpuInput_ids[i].tolist()[start[i]:end[i]+1])
                    predAnswer = ''.join(tokenizer.convert_tokens_to_string(tokens))
                    predAnswer = predAnswer.replace(' ', '')
                    EM += calc_em_score(answers[i][0], predAnswer)
                    F1 += calc_f1_score(answers[i][0], predAnswer)

        print('F1:{:.4f},EM:{:.4f}'.format(F1/count, EM/count))