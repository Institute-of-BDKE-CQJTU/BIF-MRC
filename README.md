Pytorch implementation of our paper:Few-shot Machine Reading Comprehension for Bridge Inspection via Domain-specific and Task-aware Pre-tuning Approach

Our code is based on PyTorch(1.7.1) and transformers(4.18.0), all pre-trained models are available in [huggingface](https://huggingface.co/models). Since the MRC dataset contains information about a large number of bridges in China, it is temporarily unable to open source the data

### Pre-tuning
Pre-tuning involves two steps: question type prediction and answer prediction
#### question type prediction 
```
python finalCleanQuestionGeneration/question_type_predict_train_and_test.py
```

#### answe prediction 
```
python finalCleanQuestionGeneration/answer_predict_train_and_predict.py
```

After question type prediction and answer prediction, pre-tuning data can be generated
```
python finalCleanQuestionGeneration/making_pretuning_data.py
```

Then the pre-trained model(MacBERT,Chinese_Bert_WWM,...) can be trained by pre-tuning data
```
python idea/domainPretrain.py
```


### Fine-tuning
```
python finetune/trainAndEval.py \
    --pre=True \
    --cuda=0 \
    --train_data_size=16 \
    --seed=13 \
```