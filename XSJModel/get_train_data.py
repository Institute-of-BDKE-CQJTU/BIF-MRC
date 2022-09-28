import json

datas = json.load(open('./data/preprocessingData/bridge/zong.json', 'r', encoding='UTF-8'))['data']
new_data = dict()
new_data['version'] = '1.0'
new_data['data'] = []
for data in datas:
    question = data['question']
    if '哪些' == question[:2]:
        data['location'] = 'before'
    elif '多少处' in question or '多少条' in question or '多少个' in question or '竣工时间' in question or '多少根' in question:
        data['location'] = 'middle'
    else:
        data['location'] = 'after'
    new_data['data'].append(data)

with open('./data/preprocessingData/bridge/zong_question_type.json', 'w', encoding='UTF-8') as f:
    f.write(json.dumps(new_data,ensure_ascii=False))