# seed=(43 83 271 659 859)

# domain=bridge
# cuda=${1}
# for se in 43 83 271 659 859;do
#     number=2
#     python idea/domainPretrain.py --epochs=${number} --cuda=${cuda} --seed=${se}
#     cmd="python finetune/trainAndEval.py --pre=True --cuda=${cuda} --lock=True --train_data=${domain}"
#     for((i=0;i<${#seed[*]};i++)); do
#         tempi=`expr ${i} + 1`
#         for dz in 16 32 64 128 256 512 1024 2048; do
#             ncmd="${cmd} --train_data_size=${dz} --seed=${seed[i]} --dataseed=${se}"
#             echo ${ncmd}
#             ${ncmd} &>./log/${domain}/allLock2/${se}/log${tempi}/chinese_bert_wwm_${dz}.log
#         done
#         ncmd="${cmd} --seed=${seed[i]} --full_data=True"
#         echo ${ncmd}
#         ${ncmd} &>./log/${domain}/allLock2/${se}/logfull/chinese_bert_wwm_${tempi}.log
#     done
#     echo ""
# done
seed=(43 83 271 659 859)

domain=bridge
cuda=${1}
logname=allLock_reasonable

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
for number in 100;do
    if [[ ! -d ./log/bridge/${logname}/${number} ]]
    then
        mkdir ./log/bridge/${logname}/${number}
    fi 
    for((i=0;i<${#seed[*]};i++)); do
        if [[ ! -d ./log/bridge/${logname}/${number}/${seed[i]} ]]
        then
            mkdir ./log/bridge/${logname}/${number}/${seed[i]}
        fi 
        python idea/domainPretrain.py --epochs=${number} --reasonable=True --cuda=${cuda} --seed=${seed[i]}
        cmd="python finetune/trainAndEval.py --pre=True --lock=True --cuda=${cuda} --train_data=${domain}"
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