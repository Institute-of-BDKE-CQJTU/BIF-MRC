from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn 
from torch.nn import CrossEntropyLoss
from transformers import AdamW
from tqdm import tqdm

class AnswerPredictModel(BertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.qa_outputs_before = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs_middle = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs_after = nn.Linear(config.hidden_size, config.num_labels)

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
        start_positions=None,
        end_positions=None,
        question_label=None,
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

        sequence_output = outputs[0]

        before_data_mask_index = torch.nonzero(question_label == 0).squeeze()
        middle_data_mask_index = torch.nonzero(question_label == 0).squeeze()
        after_data_mask_index = torch.nonzero(question_label == 0).squeeze()

        before_data = sequence_output.index_select(0, before_data_mask_index)
        middle_data = sequence_output.index_select(0, middle_data_mask_index)
        after_data = sequence_output.index_select(0, after_data_mask_index)

        before_logits = self.qa_outputs_before(before_data)
        before_start_logits, before_end_logits = before_logits.split(1, dim=-1)
        before_start_logits = before_start_logits.squeeze(-1).contiguous()
        before_end_logits = before_end_logits.squeeze(-1).contiguous()

        middle_logits = self.qa_outputs_middle(middle_data)
        middle_start_logits, middle_end_logits = middle_logits.split(1, dim=-1)
        middle_start_logits = middle_start_logits.squeeze(-1).contiguous()
        middle_end_logits = middle_end_logits.squeeze(-1).contiguous()

        after_logits = self.qa_outputs_after(after_data)
        after_start_logits, after_end_logits = after_logits.split(1, dim=-1)
        after_start_logits = after_start_logits.squeeze(-1).contiguous()
        after_end_logits = after_end_logits.squeeze(-1).contiguous()

        if start_positions is not None and end_positions is not None:

            before_start_positions = start_positions.index_select(0, before_data_mask_index)
            middle_start_positions = start_positions.index_select(0, middle_data_mask_index)
            after_start_positions = start_positions.index_select(0, after_data_mask_index)

            before_end_positions = end_positions.index_select(0, before_data_mask_index)
            middle_end_positions = end_positions.index_select(0, middle_data_mask_index)
            after_end_positions = end_positions.index_select(0, after_data_mask_index)

            # 计算before loss
            before_ignored_index = before_start_logits.size(1)
            before_start_positions = before_start_positions.clamp(0, before_ignored_index)
            before_end_positions = before_end_positions.clamp(0, before_ignored_index)
            
            before_loss_fct = CrossEntropyLoss(ignore_index=before_ignored_index)
            before_start_loss = before_loss_fct(before_start_logits, before_start_positions)
            before_end_loss = before_loss_fct(before_end_logits, before_end_positions)
            before_loss = (before_start_loss+before_end_loss) / 2

            # 计算middle loss
            middle_ignored_index = middle_start_logits.size(1)
            middle_start_positions = middle_start_positions.clamp(0, middle_ignored_index)
            middle_end_positions = middle_end_positions.clamp(0, middle_ignored_index)

            middle_loss_fct = CrossEntropyLoss(ignore_index=middle_ignored_index)
            middle_start_loss = middle_loss_fct(middle_start_logits, middle_start_positions)
            middle_end_loss = middle_loss_fct(middle_end_logits, middle_end_positions)
            middle_loss = (middle_start_loss+middle_end_loss) / 2

            # 计算after loss
            after_ignored_index = after_start_logits.size(1)
            after_start_positions = after_start_positions.clamp(0, after_ignored_index)
            after_end_positions = after_end_positions.clamp(0, after_ignored_index)

            after_loss_fct = CrossEntropyLoss(ignore_index=after_ignored_index)
            after_start_loss = after_loss_fct(after_start_logits, after_start_positions)
            after_end_loss = after_loss_fct(after_end_logits, after_end_positions)
            after_loss = (after_start_loss+after_end_loss) / 2

            loss = before_loss + middle_loss + after_loss

            return {'loss':loss}
        
        return {'before_start_logits':before_start_logits,
                'before_end_logits':before_end_logits,
                'middle_start_logits':middle_start_logits,
                'middle_end_logits':middle_end_logits,
                'after_start_logits':after_start_logits,
                'after_end_logits':after_end_logits
        }
    
class trainModel():

    def __init__(self, arg) -> None:
        self.lr = arg.learning_rate
        self.epochs = arg.epochs
        self.modelPath = arg.model_input_path
        self.trainData = arg.train_data
        self.cuda = int(arg.cuda)
        self.model = AnswerPredictModel.from_pretrained(self.modelPath)
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
                input_ids, attention_mask, token_type_ids, answerStart, answerEnd, question_labels = data
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, start_positions=answerStart, end_positions=answerEnd, question_label=question_labels)
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
                input_ids, attention_mask, token_type_ids, question_labels = data
                input_ids, attention_mask, token_type_ids, question_labels_cuda = input_ids.cuda(self.cuda), attention_mask.cuda(self.cuda), token_type_ids.cuda(self.cuda), question_labels.cuda(self.cuda)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, question_label=question_labels)
                before_start = outputs['before_start_logits'].cpu().argmax(dim=1)
                before_end = outputs['before_end_logits'].cpu().argmax(dim=1)
                middle_start = outputs['middle_start_logits'].cpu().argmax(dim=1)
                middle_end = outputs['middle_end_logits'].cpu().argmax(dim=1)
                after_start = outputs['after_start_logits'].cpu().argmax(dim=1)
                after_end = outputs['after_end_logits'].cpu().argmax(dim=1)
            
        # 就这里没写了

    def save(self):
        output_model_file = self.outputPath+'/'+self.trainData +'/'+str(self.cuda)+ "/modelForIdea.bin"
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        torch.save(model_to_save.state_dict(), output_model_file)
