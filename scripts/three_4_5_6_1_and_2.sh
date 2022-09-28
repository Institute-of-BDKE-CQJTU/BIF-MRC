#! /bin/bash
seed=(13 43 83 181 271 347 433 659 727 859)
model_name=(macbert roberta)
model_path=(./model/chinese_macbert_base ./model/chinese_roberta_wwm_ext)

domain=bridge
cuda=${1}
logname=ours

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

for((k=0;k<${#model_name[*]};k++)); do
    for postfix_number in 4 5 6;do
        if [[ ! -d ./log/bridge/${logname}_${postfix_number}_${model_name[k]} ]]
        then
            mkdir ./log/bridge/${logname}_${postfix_number}_${model_name[k]}
        fi 
        for number in 3;do
            if [[ ! -d ./log/bridge/${logname}_${postfix_number}_${model_name[k]}/${number} ]]
            then
                mkdir ./log/bridge/${logname}_${postfix_number}_${model_name[k]}/${number}
            fi 
            for((i=0;i<${#seed[*]};i++)); do
                if [[ ! -d ./log/bridge/${logname}_${postfix_number}_${model_name[k]}/${number}/${seed[i]} ]]
                then
                    mkdir ./log/bridge/${logname}_${postfix_number}_${model_name[k]}/${number}/${seed[i]}
                fi 
                python idea/domainPretrain.py --epochs=${number} --postfix_number=${postfix_number} --model_input_path=${model_path[k]} --cuda=${cuda} --seed=${seed[i]}
                cmd="python finetune/trainAndEval.py --pre=True --model_path=${model_path[k]} --cuda=${cuda} --train_data=${domain}"
                tempi=`expr ${i} + 1`
                for dz in 16 32 64 128 256 512 1024 2048; do
                    ncmd="${cmd} --train_data_size=${dz} --seed=${seed[i]} --dataseed=${seed[i]}"
                    echo ${ncmd}
                    ${ncmd} &>./log/${domain}/${logname}_${postfix_number}_${model_name[k]}/${number}/${seed[i]}/chinese_bert_wwm_${dz}.log
                done
                ncmd="${cmd} --seed=${seed[i]} --full_data=True"
                echo ${ncmd}
                ${ncmd} &>./log/${domain}/${logname}_${postfix_number}_${model_name[k]}/${number}/${seed[i]}/chinese_bert_wwm_${tempi}.log
            done
        done
    done
done