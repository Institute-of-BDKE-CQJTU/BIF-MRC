#! /bin/bash
seed=(43 83 271 659 859)

domain=bridge
cuda=${1}
logname=random_3808_no_postfix

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
for number in 10 20 30 40 50 60 70 80 90 100;do
    if [[ ! -d ./log/bridge/${logname}/${number} ]]
    then
        mkdir ./log/bridge/${logname}/${number}
    fi
    for((i=0;i<${#seed[*]};i++)); do
        if [[ ! -d ./log/bridge/${logname}/${number}/${seed[i]} ]]
        then
            mkdir ./log/bridge/${logname}/${number}/${seed[i]}
        fi 
        python idea/domainPretrain_temp.py --epochs=${number}  --not_postfix=True --cuda=${cuda} --seed=${seed[i]}
        cmd="python finetune/trainAndEval.py --pre=True --cuda=${cuda} --train_data=${domain}"
        tempi=`expr ${i} + 1`
        for dz in 16 32 64 128 256 512 1024 2048; do
            ncmd="${cmd} --train_data_size=${dz} --seed=${seed[i]} --dataseed=${seed[i]}"
            echo ${ncmd}
            ${ncmd} &>./log/${domain}/${logname}/${number}/${seed[i]}/chinese_bert_wwm_${dz}.log
        done
        ncmd="${cmd} --seed=${seed[i]} --full_data=True"
        echo ${ncmd}
        ${ncmd} &>./log/${domain}/${logname}/${number}/${seed[i]}/chinese_bert_wwm_${tempi}.log
    done
done
