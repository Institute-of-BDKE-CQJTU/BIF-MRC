import json

temp_dict = {0:0, 4:1, 5:2}
datas = json.load(open('./data/newQuestionGeneration/bridge/question_predict_train.json', 'r', encoding='UTF-8'))['data']
new_data = dict()
new_data['version'] = '1.0'
new_data['data'] = []
for data in datas:
    if data['postfix_index'] in [0, 4, 5]:
        data['postfix_index'] = temp_dict[data['postfix_index']]
        new_data['data'].append(data)

with open('./data/postfixDecreaseQuestionGeneration/bridge/question_predict_train.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(new_data,ensure_ascii=False))