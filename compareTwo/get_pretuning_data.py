import json

datas = json.load(open('./data/compareTwo/bridge/squad-zen-dev.json', 'r', encoding='UTF-8'))['data']

count = 0
QA_pair_data = dict()
QA_pair_data['version'] = '1.0'
QA_pair_data['data'] = []
for data in datas:
    for paragraph in data['paragraphs']:
        context = paragraph['context']
        qas = paragraph['qas']
        for qa in qas:
            if qa['is_impossible'] == True:
                continue
            question = qa['question']
            answer = qa['answers'][0]['text']
            if answer in context:
                count += 1
                temp_dict = dict()
                temp_dict['question'] = question
                temp_dict['answer'] = answer
                QA_pair_data['data'].append(temp_dict)

with open('./data/compareTwo/bridge/QA_pair.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(QA_pair_data,ensure_ascii=False))