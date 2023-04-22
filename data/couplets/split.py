import json
import re

sample_pair = []

with open('./train/in.txt', 'r') as f:
    lines = f.readlines()
    counter = 0
    for data_line in lines:
        data_line = data_line.replace(' ', '')
        data_line = data_line.replace('\n', '')
        #print(data_line)
        counter += 1
        sample_pair.append([data_line])

train_size = counter
#print("sample pair num: ", train_size)

with open('./train/out.txt', 'r') as f:
    lines = f.readlines()
    counter = 0
    for data_line in lines:
        data_line = data_line.replace(' ', '')
        data_line = data_line.replace('\n', '')
        sample_pair[counter].append(data_line)
        counter += 1

with open('./test/in.txt', 'r') as f:
    lines = f.readlines()
    counter = train_size
    for data_line in lines:
        data_line = data_line.replace(' ', '')
        data_line = data_line.replace('\n', '')
        #print(data_line)
        counter += 1
        sample_pair.append([data_line])

print("sample pair num: ", counter)

with open('./test/out.txt', 'r') as f:
    lines = f.readlines()
    counter = train_size
    for data_line in lines:
        data_line = data_line.replace(' ', '')
        data_line = data_line.replace('\n', '')
        sample_pair[counter].append(data_line)
        counter += 1

#### 检验
errors = 0
for pair in sample_pair:
    if len(pair)!=2:
        errors += 1

print(errors, sample_pair[0])

### 构造对齐的数据对
couplet_data = []

for pair in sample_pair:
    couplet_data.append({"instruction": pair[0], "input": "", "output": pair[1]})

with open("couplet-all"+'.json', 'w', encoding='utf-8') as f:
    json.dump(couplet_data, f, ensure_ascii=False, indent=1)
