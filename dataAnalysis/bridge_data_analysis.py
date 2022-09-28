import json

questionPostfix = ['是多少？', '位于哪里？', '问题？', '的原因？', '是什么？', '维修建议？']
datas = json.load(open('./data/preprocessingData/bridge/train.json', 'r', encoding='UTF-8'))['data']
length, count = 0.0, 0
postfix_count = 0
question_type_number = [0]*6
for data in datas:
    question = data['question']
    count += 1
    for i in range(6):
        if questionPostfix[i] in question:
            question_type_number[i] += 1
            postfix_count += 1
            break
print(question_type_number)
print(postfix_count, count)
# [635, 121, 38, 65, 682, 339]
# 1880 2844
# print(length / count)
# 答案平均长度为12.6