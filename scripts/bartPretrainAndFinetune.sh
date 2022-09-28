#! /bin/bash
seed=(13 43 83 181 271 347 433 659 727 859)

domain=bridge
cuda=${1}
logname=ours

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

# for postfix_number in 4 5 6;do 
for postfix_number in 6;do 
    if [[ ! -d ./log/bridge/${logname}_${postfix_number}_bart ]]
    then
        mkdir ./log/bridge/${logname}_${postfix_number}_bart
    fi 
    for number in 1 2;do
        if [[ ! -d ./log/bridge/${logname}_${postfix_number}_bart/${number} ]]
        then
            mkdir ./log/bridge/${logname}_${postfix_number}_bart/${number}
        fi 
        for((i=0;i<${#seed[*]};i++)); do
            if [[ ! -d ./log/bridge/${logname}_${postfix_number}_bart/${number}/${seed[i]} ]]
            then
                mkdir ./log/bridge/${logname}_${postfix_number}_bart/${number}/${seed[i]}
            fi 
            python bart/domainPretrain.py --epochs=${number} --postfix_number=${postfix_number} --cuda=${cuda} --seed=${seed[i]}
            cmd="python bart/trainAndEval.py --pre=True --cuda=${cuda} --train_data=${domain}"
            tempi=`expr ${i} + 1`
            for dz in 16 32 64 128 256 512 1024 2048; do
                ncmd="${cmd} --train_data_size=${dz} --seed=${seed[i]} --dataseed=${seed[i]}"
                echo ${ncmd}
                ${ncmd} &>./log/${domain}/${logname}_${postfix_number}_bart/${number}/${seed[i]}/bart_${dz}.log
            done
            ncmd="${cmd} --seed=${seed[i]} --full_data=True"
            echo ${ncmd}
            ${ncmd} &>./log/${domain}/${logname}_${postfix_number}_bart/${number}/${seed[i]}/bart_${tempi}.log
        done
    done
done


# logname=bart_finetune
# if [[ ! -d ./log/bridge/${logname} ]]
# then
#     mkdir ./log/bridge/${logname}
# fi
# for number in 20;do
#     if [[ ! -d ./log/bridge/${logname}/${number} ]]
#     then
#         mkdir ./log/bridge/${logname}/${number}
#     fi  
#     cmd="python bart/trainAndEval.py --train_data=${domain} --cuda=${cuda}"
#     for((i=0;i<${#seed[*]};i++)); do
#         if [[ ! -d ./log/bridge/${logname}/${number}/${seed[i]} ]]
#         then
#             mkdir ./log/bridge/${logname}/${number}/${seed[i]}
#         fi 
#         tempi=`expr ${i} + 1`
#         for dz in 16 32 64 128 256 512 1024 2048; do
#             ncmd="${cmd} --train_data_size=${dz} --seed=${seed[i]} --dataseed=${seed[i]}"
#             echo ${ncmd}
#             ${ncmd} &>./log/${domain}/${logname}/${number}/${seed[i]}/bart_${dz}.log
#         done
#         ncmd="${cmd} --seed=${seed[i]} --full_data=True"
#         echo ${ncmd}
#         ${ncmd} &>./log/${domain}/${logname}/${number}/${seed[i]}/bart_${tempi}.log
#     done
# done