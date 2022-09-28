from transformers import BartTokenizer, BartForQuestionAnswering
import torch

# tokenizer = BartTokenizer.from_pretrained("model/bart-base-chinese")
model = BartForQuestionAnswering.from_pretrained("./model/bart-base-chinese")
for name ,param in model.named_parameters():
    print(name)