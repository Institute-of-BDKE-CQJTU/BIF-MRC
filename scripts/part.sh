#! /bin/bash
seed=(13 43 83 181 271 347 433 659 727 859)

domain=bridge
cuda=${1}
logname=part

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
for part in 0.2 0.4 0.6 0.8;do
    if [[ ! -d ./log/bridge/${logname}/${part} ]]
    then
        mkdir ./log/bridge/${logname}/${part}
    fi
    for number in 20;do
        if [[ ! -d ./log/bridge/${logname}/${part}/${number} ]]
        then
            mkdir ./log/bridge/${logname}/${part}/${number}
        fi 
        for((i=0;i<${#seed[*]};i++)); do
            if [[ ! -d ./log/bridge/${logname}/${part}/${number}/${seed[i]} ]]
            then
                mkdir ./log/bridge/${logname}/${part}/${number}/${seed[i]}
            fi 
            python idea/domainPretrain.py --epochs=${number} --reasonable=True --cuda=${cuda}  --part=${part} --seed=${seed[i]}
            cmd="python finetune/trainAndEval.py --pre=True --cuda=${cuda} --train_data=${domain}"
            tempi=`expr ${i} + 1`
            for dz in 16 32 64 128 256 512 1024 2048; do
                ncmd="${cmd} --train_data_size=${dz} --seed=${seed[i]} --dataseed=${seed[i]}"
                echo ${ncmd}
                ${ncmd} &>./log/${domain}/${logname}/${part}/${number}/${seed[i]}/chinese_bert_wwm_${dz}.log
            done
            ncmd="${cmd} --seed=${seed[i]} --full_data=True"
            echo ${ncmd}
            ${ncmd} &>./log/${domain}/${logname}/${part}/${number}/${seed[i]}/chinese_bert_wwm_${tempi}.log
        done
    done
done