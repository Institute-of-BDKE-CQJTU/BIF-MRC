import torch
import argparse
import sys
sys.path.append('./')
from modeling import trainModel
from prepareData.data import trainDataset, devDataset, getDevDataLoader, getTrainDataLoader

def main(arg):
    torch.manual_seed(arg.seed)
    torch.cuda.manual_seed_all(arg.seed)
    trainset = trainDataset('./data/dataFeatures/'+arg.train_data+'/train-'+str(arg.train_data_size)+'-'+str(arg.dataseed)+'.pkl')
    if arg.pre == True:
        trainset = trainDataset('./data/dataFeatures/'+arg.train_data+'/train-'+str(arg.train_data_size)+'-'+str(arg.dataseed)+'.pkl')
    if arg.full_data == True:
        trainset = trainDataset('./data/dataFeatures/'+arg.train_data+'/train.pkl')
    devset = devDataset('./data/dataFeatures/'+arg.train_data+'/dev.pkl')
    trainDataLoader = getTrainDataLoader(trainset, batch_size=arg.batch_size)
    devDataLoader = getDevDataLoader(devset, batch_size=arg.batch_size)
    trainer = trainModel(arg)
    trainer.train(trainDataLoader)
    trainer.test(devDataLoader)
    if arg.save == True:
        trainer.save('./finetune/')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--model_path', type=str, default='./model/chinese_bert_wwm', help='模型选择')
    parse.add_argument('--batch_size', type=int, default=20, help='batch_size')
    parse.add_argument('--train_data', type=str, default='bridge')
    parse.add_argument('--train_data_size', type=int, default=16)
    parse.add_argument('--full_data', type=bool, default=False)
    parse.add_argument('--epochs', type=int, default=10)
    parse.add_argument('--learning_rate', type=float, default=1e-5)
    parse.add_argument('--seed', type=int, default=43)
    parse.add_argument('--dataseed', type=int, default=43)
    parse.add_argument('--cuda', type=int, default=0)
    parse.add_argument('--pre', type=bool, default=False)
    parse.add_argument('--lock', type=bool, default=False)
    parse.add_argument('--save', type=bool, default=False)

    arg = parse.parse_args()
    main(arg)