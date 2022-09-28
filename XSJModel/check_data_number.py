import json

datas = json.load(open('./data/preprocessingData/bridge/zong_question_type.json', 'r', encoding='UTF-8'))['data']

numbers = [0]*3
index_dict = {'before':0, 'middle':1, 'after':2}
for data in datas:
    numbers[index_dict[data['location']]] += 1
print(numbers)
for i in range(3):
    print(numbers[i] / sum(numbers))