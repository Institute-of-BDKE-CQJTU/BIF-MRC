from transformers import BertTokenizer, BertForPretrainedModel, BertModel, AdamW
from torch import nn
import torch
from torch.nn import CrossEntropyLoss, BCELoss, MSELoss, BCEWithLogitsLoss
import pickle
from tqdm import tqdm
import argparse
from torch.utils.data import TensorDataset, DataLoader
from finetune.evaluate import *
from sklearn.metrics import accuracy_score, recall_score, f1_score

tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')

class EndToEndModel(BertForPretrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

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
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
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
        sequence_output = outputs[0]

        qa_logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = qa_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        label_logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(label_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(label_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(label_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(label_logits, labels)

        total_loss = total_loss + loss
        return {'total_loss':total_loss,
                'label_logits':label_logits,
                'start_logits':start_logits,
                'end_logits':end_logits
        }


class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.model = EndToEndModel.from_pretrained(self.modelPath, num_labels=4)
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
                input_ids, attention_mask, token_type_ids, labels, answer_start, answer_end = data
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, start_positions=answer_start, end_positions=answer_end)
                loss = outputs['total_loss']
                loop.set_postfix(loss = loss.item())
                loss.backward()
                self.optim.step()
    
    def test(self, test_loader):
        self.model.eval()
        predict_type, answer_start, answer_end = [], [], []
        with torch.no_grad():
            for iter, data in enumerate(test_loader):
                input_ids, attention_mask, token_type_ids = data
                input_ids_gpu, attention_mask, token_type_ids = input_ids.cuda(self.cuda), attention_mask.cuda(self.cuda), token_type_ids.cuda(self.cuda)
                outputs = self.model(input_ids=input_ids_gpu, attention_mask=attention_mask, token_type_ids=token_type_ids)
                classification_type = outputs['label_logits'].cpu().argmax(1).tolist()
                predict_type.extend(classification_type)
                start = outputs['start_logits'].cpu().argmax(dim=1)
                end = outputs['end_logits'].cpu().argmax(dim=1)
                answer_start.extend(start)
                answer_end.extend(end)
        return predict_type, answer_start, answer_end

    def get_result_socre(self, answers, labels, predict_type, answer_start, answer_end, input_tokens):
        print('accuracy:', accuracy_score(labels, predict_type))
        print('recall:', recall_score(labels, predict_type, average='micro'))
        print('f1 score:', f1_score(labels, predict_type, average='micro'))
        EM, F1 = 0.0, 0.0
        for i in range(len(answers)):
            predict_answer = tokenizer.convert_tokens_to_string(input_tokens[answer_start[i]:answer_end[i]+1])
            predict_answer = predict_answer.replace(' ', '')
            EM += calc_em_score(answers[i], predict_answer)
            F1 += calc_f1_score(answers[i], predict_answer)
        print('EM:', EM/len(answers))
        print('F1:', F1/len(answers))


    def save(self):
        output_model_file = self.outputPath+'/'+self.trainData +"/"+str(self.cuda)+ "/modelForIdea.bin"
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), output_model_file)

def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    f = open('./data/newQuestionGeneration/bridge/end_to_end_train.pkl', 'rb')
    features = pickle.load(f)
    length = len(features)
    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[:int(length*arg.part),:]
    labels = torch.tensor([f[3] for f in features], dtype=torch.long)[:int(length*arg.part)]
    answerStart = torch.tensor([f[4] for f in features], dtype=torch.long)[:int(length*arg.part)]
    answerEnd = torch.tensor([f[5] for f in features], dtype=torch.long)[:int(length*arg.part)]
    trainDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids, labels, answerStart, answerEnd)
    trainDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=True)
    f.close()
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)

    all_input_ids = torch.tensor([f[0] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    all_input_mask = torch.tensor([f[1] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    all_label_ids = torch.tensor([f[2] for f in features], dtype=torch.long)[int(length*arg.part):,:]
    labels = torch.tensor([f[3] for f in features], dtype=torch.long)[int(length*arg.part):].tolist()
    answers = [f[6] for f in features]
    input_tokens = [f[7] for f in features]
    testDataset = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    testDataLoader = DataLoader(trainDataset, batch_size=arg.batch_size, shuffle=False)
    predict_type, answer_start, answer_end = trainer.test(testDataLoader)
    trainer.get_result_socre(answers, labels, predict_type, answer_start, answer_end, input_tokens)
    # trainer.save()

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument('--model_input_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    parse.add_argument('--model_output_path', type=str, default='./newQuestionGeneration/model', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--epochs', type=int, default=1)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=42)
    parse.add_argument('--part', type=float, default=0.8)
    parse.add_argument('--cuda', type=int, default=1)

    arg = parse.parse_args()
    main(arg)