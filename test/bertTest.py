# from transformers import BertTokenizer, BertForQuestionAnswering
# import torch

# tokenizer = BertTokenizer.from_pretrained('./model/chinese_bert_wwm')
# model = BertForQuestionAnswering.from_pretrained('./model/chinese_bert_wwm')
# for param in model.named_parameters():
#     if str(param[0])[:4] != 'bert':
#         print(param[0])
    # p.requires_grad=False
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(device)
# model.to(device)
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("model/chinese_bert_wwm")
model = BertForSequenceClassification.from_pretrained("model/chinese_bert_wwm")

inputs = tokenizer("Hello, my dog is cute", "here is my dog", return_tensors="pt")
print(inputs)