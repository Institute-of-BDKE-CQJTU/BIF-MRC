"""
    该文件负责实现dataloader和dataset
"""
from re import search
import torch
import pickle
from torch.utils.data import Dataset, DataLoader

class trainDataset(Dataset):
    def __init__(self, path:str) -> None:
        super().__init__()
        with open(path, 'rb') as f:
            self.features = pickle.load(f)
    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids, attention_mask, token_type_ids, answerStart, answerEnd = feature
        return input_ids, attention_mask, token_type_ids, answerStart, answerEnd

def trainCollateFunc(features):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    answerStart = []
    answerEnd = []
    for feature in features:
        input_ids.append(feature[0])
        attention_mask.append(feature[1])
        token_type_ids.append(feature[2])
        answerStart.append([feature[3]])
        answerEnd.append([feature[4]])
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)
    answerStart = torch.tensor(answerStart)
    answerEnd = torch.tensor(answerEnd)

    return input_ids, attention_mask, token_type_ids, answerStart, answerEnd


def getTrainDataLoader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=trainCollateFunc, shuffle=True)

class devDataset(Dataset):
    def __init__(self, path:str) -> None:
        super().__init__()
        with open(path, 'rb') as f:
            self.features = pickle.load(f)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        input_ids, attention_mask, token_type_ids, answers = feature
        return input_ids, attention_mask, token_type_ids, answers

def devCollateFunc(features):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    answers = []
    for feature in features:
        input_ids.append(feature[0])
        attention_mask.append(feature[1])
        token_type_ids.append(feature[2])
        answers.append([feature[3]])
    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    token_type_ids = torch.tensor(token_type_ids)

    return input_ids, attention_mask, token_type_ids, answers

def getDevDataLoader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=devCollateFunc, shuffle=True)

