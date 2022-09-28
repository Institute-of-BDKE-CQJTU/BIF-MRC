import json
from transformers import  BertModel, BertConfig, BertPreTrainedModel, BertTokenizer, AdamW
from torch import nn
from torch.nn import MSELoss
import torch
import argparse
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import pickle
from tqdm import tqdm

class AnswerLengthPredict(BertPreTrainedModel):

    def __init__(self, config) -> None:
        super().__init__(config)
        
        self.bert = BertModel(config, add_pooling_layer=True)

        self.length_predict_layer = nn.Linear(config.hidden_size, 1)

        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        length=None,
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
        answer_length = self.length_predict_layer(pooled_output).squeeze(-1).contiguous()
        loss_func = MSELoss()
        loss = loss_func(answer_length, length)
        return loss, answer_length

class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.model = AnswerLengthPredict.from_pretrained(self.modelPath)
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
                input_ids, attention_mask, token_type_ids, answer_length = data
                loss, _ = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, length=answer_length)
                loop.set_postfix(loss = loss.item())
                loss.backward()
                self.optim.step()
    
    def test(self, test_loader):
        self.model.eval()
        predict_length = []
        with torch.no_grad():
            for iter, data in enumerate(test_loader):
                input_ids, attention_mask, token_type_ids = data
                input_ids, attention_mask, token_type_ids = input_ids.cuda(self.cuda), attention_mask.cuda(self.cuda), token_type_ids.cuda(self.cuda)
                _, length = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                length = length.cpu().tolist()
                predict_length.extend(length)
        return predict_length

    def save(self):
        output_model_file = self.outputPath+'/'+self.trainData +"/"+str(self.cuda)+ "/modelForIdea.bin"
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), output_model_file)

def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/questionData/bridge/answer_length_train.pkl', 'rb')
    features = pickle.load(f)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)
    length = torch.tensor([f[3] for f in features], dtype=torch.long)
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, length)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size)
    f.close()
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    trainer.save()

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument('--model_input_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./questionGeneration/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=1)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--part', type=float, default=1)
    parse.add_argument('--cuda', type=int, default=1)
    parse.add_argument('--not_postfix', type=bool, default=False)

    arg = parse.parse_args()
    main(arg)



