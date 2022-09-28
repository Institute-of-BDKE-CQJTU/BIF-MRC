import json
import re
import random

def extractLongLine(lineLength:int):
    fin = open('./data/domainData/bridge/clean.txt', 'w', encoding='UTF-8')
    with open('./data/domainData/bridge/raw.txt', 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            if len(line) >= lineLength:
                fin.write(line)
                fin.write('\n')
    fin.close()


def writeConstructData(contexts, questions, answers):
    newData = dict()
    newData['version'] = '1.0'
    newData['data'] = []
    for i in range(len(answers)):
        pairData = dict()
        pairData['context'] = contexts[i]
        pairData['question'] = questions[i]
        pairData['answer'] = answers[i]
        newData['data'].append(pairData)
    with open('./data/finalCleanQuestionGeneration/bridge/construct_random_data_no_postfix.json', 'w', encoding='UTF-8') as f:
        f.write(json.dumps(newData,ensure_ascii=False))

# idea1
# spanA?context[spanA[spanB]]
# def constructData(longAnswer:int, shortAnswer:int):
def constructData():
    lengths = [5, 15, 11, 33, 17, 19]
    questionShortPostfix = ['是多少？', '位于哪里？', '问题？']
    questionLongPostfix = ['的原因？', '是什么？', '维修建议？']
    questions = []
    contexts = []
    answers = []
    with open('./data/domainData/bridge/clean.txt', 'r', encoding='UTF-8') as f:
        for line in f:
            templine = line.strip()
            line = templine.split('。')
            for sentence in line:
                try:
                    if len(sentence) <= 10:
                        continue
                    number = random.randint(0, 5)
                    questionPostfix = ''
                    if number >= 3:
                        questionPostfix = questionLongPostfix[number-3]
                    else:
                        questionPostfix = questionShortPostfix[number]
                    answerLength = random.randint(lengths[number]-2, lengths[number]+2)
                    answer_start = random.randint(1, len(sentence)-answerLength)
                    answer = sentence[answer_start:answer_start+answerLength]
                    j = 0
                    for j in range(1, answer_start):
                        if sentence[answer_start-j] in [',', '，', ':', '：', '.', '。', '、']:
                            break
                    question = sentence[answer_start - j+1:answer_start]+questionPostfix
                    # question = sentence[:len(sentence)-answerLength] + questionPostfix
                except:
                    continue
                questions.append(question)
                answers.append(answer)
                # 这里之前写错了，写成sentence了，应该写line
                # contexts.append(sentence)
                contexts.append(templine)
    # return contexts, questions, answers
    writeConstructData(contexts, questions, answers)

def constructData2():
    lengths = [5, 15, 11, 33, 17, 19]
    questionShortPostfix = ['是多少？', '位于哪里？', '问题？']
    questionLongPostfix = ['的原因？', '是什么？', '维修建议？']
    questions = []
    contexts = []
    answers = []
    with open('./data/domainData/bridge/clean.txt', 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip().split('。')
            for sentence in line:
                try:
                    if len(sentence) <= 10:
                        continue
                    number = random.randint(0, 5)
                    questionPostfix = ''
                    if number >= 3:
                        questionPostfix = questionLongPostfix[number-3]
                    else:
                        questionPostfix = questionShortPostfix[number]
                    answerLength = random.randint(lengths[number]-2, lengths[number]+2)
                    answer_start = random.randint(1, len(sentence)-answerLength)
                    answer = sentence[answer_start:answer_start+answerLength]
                    j = 0
                    for j in range(1, answer_start):
                        if sentence[answer_start-j] in [',', '，', ':', '：', '.', '。', '、']:
                            break
                    question = sentence[answer_start - j+1:answer_start]+questionPostfix
                    # question = sentence[:len(sentence)-answerLength] + questionPostfix
                except:
                    continue
                questions.append(question)
                answers.append(answer)
                # 这里之前写错了，写成sentence了，应该写line
                contexts.append(sentence)
    # return contexts, questions, answers
    writeConstructData(contexts, questions, answers)

def construct_random_data():
    questionShortPostfix = ['是多少？']
    questionLongPostfix = ['是什么？', '维修建议？']
    lengths = [5, 17, 19]
    questions = []
    contexts = []
    answers = []
    with open('./data/finalCleanQuestionGeneration/bridge/clean.txt', 'r', encoding='UTF-8') as f:
        for line in f:
            line = line.strip()
            sentences = line.split('。')
            for sentence in sentences:
                try:
                    if len(sentence) <= 8:
                        continue
                    number = random.randint(0, 2)
                    questionPostfix = ''
                    if number >= 1:
                        questionPostfix = questionLongPostfix[number-1]
                    else:
                        questionPostfix = questionShortPostfix[number]
                    answerLength = random.randint(lengths[number]-2, lengths[number]+2)
                    answer_start = random.randint(1, len(sentence)-answerLength)
                    answer = sentence[answer_start:answer_start+answerLength]
                    j = 0
                    for j in range(1, answer_start):
                        if sentence[answer_start-j] in [',', '，', ':', '：', '.', '。', '、']:
                            break
                    # question = sentence[answer_start - j+1:answer_start]+questionPostfix
                    question = sentence[answer_start - j+1:answer_start]+'？'
                    questions.append(question)
                    answers.append(answer)
                    contexts.append(line)
                except:
                    continue
    writeConstructData(contexts, questions, answers)

if __name__ == "__main__":
    # extractLongLine(400)
    # constructData()
    construct_random_data()