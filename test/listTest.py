file=open('./test/pretuning_index2.txt','r')
pretuning_index_str = file.readlines()[0].strip()[1:-1]
pretuning_index = list(map(int, pretuning_index_str.split(',')))
file.close()

file=open('./test/origin_index2.txt','r')
origin_index_str = file.readlines()[0].strip()[1:-1]
origin_index = list(map(int, origin_index_str.split(',')))
file.close()

newlist = pretuning_index
for i in range(len(origin_index)):
    if origin_index[i] not in pretuning_index:
        newlist.append(origin_index[i])

file=open('./test/all_drop_index.txt','w')  
file.write(str(newlist))
file.close()