from transformers import AlbertForQuestionAnswering, AlbertForSequenceClassification, AdamW, BertForSequenceClassification, BertTokenizer
import torch
import argparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score
import json
from transformers import BertModel, BertPreTrainedModel
from torch import nn 
from focal_loss import *


class MyClassificationModel(BertPreTrainedModel):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if labels == None:
            return {'logits':logits}

        loss_fct = MultiFocalLoss(num_class=self.config.num_labels, alpha=[0.87, 0.11, 0.01])
        loss = loss_fct(logits, labels)
        return {'loss':loss, 'logits':logits}

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.cuda = int(arg.cuda)
        self.model = MyClassificationModel.from_pretrained(self.modelPath, num_labels=3)
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
                loss = outputs['loss']
                loop.set_postfix(loss_value = loss.item())
                loss.backward()
                self.optim.step()

    def test(self, test_loader):
        self.model.eval()
        predict_type = []
        with torch.no_grad():
            loop = tqdm(enumerate(test_loader), total = len(test_loader))
            for iter, data in loop:
                loop.set_description(f'Test')
                input_ids, attention_mask, token_type_ids = data
                input_ids, attention_mask, token_type_ids = input_ids.cuda(self.cuda), attention_mask.cuda(self.cuda), token_type_ids.cuda(self.cuda)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                classification_type = outputs['logits'].cpu().argmax(1).tolist()
                predict_type.extend(classification_type)
        return predict_type

    def eval(self, predict_type, eval_type):
        print('accuracy:', accuracy_score(eval_type, predict_type))
        print('recall:', recall_score(eval_type, predict_type, average='micro'))
        print('f1 score:', f1_score(eval_type, predict_type, average='micro'))

    def save(self):
        output_model_file = self.outputPath+'/'+self.trainData +'/'+str(self.cuda)+ "/modelForIdea.bin"
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), output_model_file)

        
def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/XSJModel/bridge/question_type_train.pkl', 'rb')
    features = pickle.load(f)
    # train
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[:int(length*arg.part), :]
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[:int(length*arg.part), :]
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[:int(length*arg.part), :]
    labels = torch.tensor([f[3] for f in features], dtype=torch.long)[:int(length*arg.part)]
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, labels)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    # test
    f.close()

    f = open('./data/XSJModel/bridge/question_type_train.pkl', 'rb')
    features = pickle.load(f)
    # train
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[int(length*arg.part):, :]
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[int(length*arg.part):, :]
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[int(length*arg.part):, :]
    labels = torch.tensor([f[3] for f in features], dtype=torch.long)[int(length*arg.part):].tolist()
    testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    testDataLoader = DataLoader(testDataset, batch_size=arg.batch_size, shuffle=False)
    predict_type = trainer.test(testDataLoader)
    trainer.eval(predict_type, labels)
    f.close()
    file=open('./XSJModel/question_postfix.txt','w')  
    file.write(str(predict_type))
    file.close()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    # parse.add_argument('--model_input_path', type=str, default='./model/albert_chinese_small', help='模型选择')
    # parse.add_argument('--model_input_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    parse.add_argument('--model_input_path', type=str, default='./model/chinese_roberta_wwm_ext', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./postfixDecreaseQuestionGeneration/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=1)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--cuda', type=int, default=1)
    parse.add_argument('--part', type=float, default=0.8)

    arg = parse.parse_args()
    main(arg)