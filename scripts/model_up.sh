#! /bin/bash
seed=(43 83 271 659 859)

domain=bridge
cuda=${1}
logname=model_up

if [[ -z ${cuda} ]]
then 
    echo "未输入GPU ID"
    exit 0
fi

if [[ -z ${logname} ]]
then 
    echo "未输入日志文件名"
    exit 0
fi

if [[ ! -d ./log/bridge/${logname} ]]
then
    mkdir ./log/bridge/${logname}
fi

for((i=0;i<${#seed[*]};i++)); do
    python idea/domainPretrain.py --epochs=100 --cuda=${cuda} --seed=${seed[i]}
    python postfixDecreaseQuestionGeneration/model_up.py --epochs=20 --seed=${seed[i]} --cuda=${cuda} &>./log/${domain}/${logname}/chinese_bert_wwm_${seed[i]}.log
done