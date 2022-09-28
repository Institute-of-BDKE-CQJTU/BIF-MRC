import json

datas = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']

length = [0, 0, 0, 0, 0, 0]
number = [0, 0, 0, 0, 0, 0]
questionPostfix = ['多少？', '哪里？', '什么？', '原因？', '建议？', '问题？']
for data in datas:
    question = data['question']
    answer = data['answer']
    for i in range(6):
        if question[-3:] == questionPostfix[i]:
            length[i] += len(answer)
            number[i] += 1
for i in range(6):
    length[i] = length[i] / number[i]
print(length)
print(number)