import json
import json_lines

path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/chihuixuan/llm/alpaca-lora/firefly/firefly-train-1.1M.jsonl'
#sample_num = 2000
total_samples = 0
kind_map = {} # 各个kind的语料总数
kind_counter = {} #各个kind的当前计数器
kind_split_point = {} # 各个kind按5段切分的每段数量

split_data5 = [[], [], [], [], []] # 5份切分data

data = []
with open(path, 'rb') as f: 
   for item in json_lines.reader(f):
       data.append(item)

print('data load done!')
#print(type(data[0]), data[0]['kind'])

#########
# 确定各个kind的语料总数，初始化计数器
for data_line in data:
    if data_line['kind'] not in kind_map.keys():
        kind_map[data_line['kind']] = 1
        kind_counter[data_line['kind']] = 0
    else:
        kind_map[data_line['kind']] += 1
    total_samples += 1

for key in kind_map.keys():
    if kind_map[key]%5 == 0:
        kind_split_point[key] = kind_map[key]//5
    else:
        kind_split_point[key] = kind_map[key]//5 + 1

print("key count of each kind: ", kind_map)

### 开始切分
for data_line in data:
    now_kind = data_line['kind']
    position = kind_counter[now_kind]//kind_split_point[now_kind]
    kind_counter[now_kind] += 1
    split_data5[position].append(data_line)

########################### 验证切分是否正确, 此段代码用来check，可去掉
evalution_total = {}
eval_samples = 0
# 总数验证
for i in range(5):
    eval_samples += len(split_data5[i])
    for data_line in split_data5[i]:
        if data_line['kind'] not in evalution_total.keys():
            evalution_total[data_line['kind']] = 1
        else:
            evalution_total[data_line['kind']] += 1

# check 整体数量
if eval_samples == total_samples:
    print("total samples correct!!!")
else:
    print("### total samples INCORRECT!!!")

# check 各类别内部总数
for key in evalution_total.keys():
    if evalution_total[key] != kind_map[key]:
        print("### " + key + " INCORRECT!!!")
#####################################################################

split_data5_saved = [[], [], [], [], []]
for i in range(5):
    for data_line in split_data5[i]:
        split_data5_saved[i].append({"instruction": data_line["input"], "input": "", "output": data_line["target"]})
    with open("firefly-train-"+str(i)+'.json', 'w', encoding='utf-8') as f:
        json.dump(split_data5_saved[i], f, ensure_ascii=False, indent=1)
