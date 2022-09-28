#! /bin/bash
seed=(13 43 83 181 271 347 433 659 727 859)

domain=bridge
cuda=${1}
logname=macbert

if [[ -z ${1} ]]
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
for number in 20;do
    if [[ ! -d ./log/bridge/${logname}/${number} ]]
    then
        mkdir ./log/bridge/${logname}/${number}
    fi 
    for((i=0;i<${#seed[*]};i++)); do
        if [[ ! -d ./log/bridge/${logname}/${number}/${seed[i]} ]]
        then
            mkdir ./log/bridge/${logname}/${number}/${seed[i]}
        fi 
        python idea/domainPretrain.py --epochs=${number} --model_input_path=./model/chinese_macbert_base --reasonable=True --cuda=${cuda} --seed=${seed[i]}
        cmd="python finetune/trainAndEval.py --pre=True --cuda=${cuda} --train_data=${domain} --model_path=./model/chinese_macbert_base "
        tempi=`expr ${i} + 1`
        for dz in 16 32 64 128 256 512 1024 2048; do
            ncmd="${cmd} --train_data_size=${dz} --seed=${seed[i]} --dataseed=${seed[i]}"
            echo ${ncmd}
            ${ncmd} &>./log/${domain}/${logname}/${number}/${seed[i]}/macbert_${dz}.log
        done
        ncmd="${cmd} --seed=${seed[i]} --full_data=True"
        echo ${ncmd}
        ${ncmd} &>./log/${domain}/${logname}/${number}/${seed[i]}/macbert_${tempi}.log
    done
done


logname=macbert_finetune
if [[ ! -d ./log/bridge/${logname} ]]
then
    mkdir ./log/bridge/${logname}
fi
for number in 20;do
    if [[ ! -d ./log/bridge/${logname}/${number} ]]
    then
        mkdir ./log/bridge/${logname}/${number}
    fi 
    for((i=0;i<${#seed[*]};i++)); do
        if [[ ! -d ./log/bridge/${logname}/${number}/${seed[i]} ]]
        then
            mkdir ./log/bridge/${logname}/${number}/${seed[i]}
        fi 
        cmd="python finetune/trainAndEval.py --cuda=${cuda} --train_data=${domain} --model_path=./model/chinese_macbert_base "
        tempi=`expr ${i} + 1`
        for dz in 16 32 64 128 256 512 1024 2048; do
            ncmd="${cmd} --train_data_size=${dz} --seed=${seed[i]} --dataseed=${seed[i]}"
            echo ${ncmd}
            ${ncmd} &>./log/${domain}/${logname}/${number}/${seed[i]}/macbert_${dz}.log
        done
        ncmd="${cmd} --seed=${seed[i]} --full_data=True"
        echo ${ncmd}
        ${ncmd} &>./log/${domain}/${logname}/${number}/${seed[i]}/macbert_${tempi}.log
    done
done