"""
对训练数据进行16-32-64-...-2048采样
"""
import random
import json

# random.seed(42)

# data = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']

# counts = [16, 32, 64, 128, 256, 512, 1024, 2048]

# for count in counts:
#     newData = dict()
#     newData['version'] = '1.0'
#     newData['data'] = []
#     sampleNumber = []
#     for i in range(count):
#         sampleNumber.append(random.randint(0, 2736))
#     for number in sampleNumber:
#         if data[number]['answer'] != '':
#             newData['data'].append(data[number])
#         else:
#             newData['data'].append(data[number+1])
#     with open('./data/sampleData/bridge/train-'+str(count)+'.json', 'w', encoding='UTF-8') as f:
#         f.write(json.dumps(newData,ensure_ascii=False))

seeds = [13, 43, 83, 181, 271, 347, 433, 659, 727, 859]
for seed in seeds:
    random.seed(seed)

    data = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']

    counts = [16, 32, 64, 128, 256, 512, 1024, 2048]

    for count in counts:
        newData = dict()
        newData['version'] = '1.0'
        newData['data'] = []
        sampleNumber = []
        for i in range(count):
            sampleNumber.append(random.randint(0, 2736))
        for number in sampleNumber:
            if data[number]['answer'] != '':
                newData['data'].append(data[number])
            else:
                newData['data'].append(data[number+1])
        with open('./data/sampleData/bridge/train-'+str(count)+'-'+str(seed)+'.json', 'w', encoding='UTF-8') as f:
            f.write(json.dumps(newData,ensure_ascii=False))