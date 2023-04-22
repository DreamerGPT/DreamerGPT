import json
import re

with open('PsyQA_full.json','r',encoding='utf-8') as f:
    json_data = json.load(f)

# 1个question（作为instruction）对应多个answer（作为output），不考虑description ==> psyqa-0.json
# 1个question（作为instruction）只对应其中的一个answer（作为output），不考虑description ==> psyqa-1.json
# 1个question（作为instruction）对应多个answer（作为output），考虑description，把其文本拼接到question后面共同作为instruction ==> psyqa-2.json
# 1个question（作为instruction）只对应其中的一个answer（作为output），考虑description，把其文本拼接到question后面共同作为instruction  ==> psyqa-3.json
# 1个question（作为instruction）对应多个answer（作为output），考虑description（作为input） ==> psyqa-4.json
# 1个question（作为instruction）只对应其中的一个answer（作为output），考虑description（作为input） ==> psyqa-5.json

split_data = [[],[],[],[],[],[]]

for data_line in json_data:
    question = data_line['question']
    description = data_line['description']
    answer_count = 0
    for answer in data_line['answers']:
        answer_text = answer['answer_text']
        if answer_count == 0:
            split_data[0].append({"instruction": question, "input": "", "output": answer_text})
            split_data[1].append({"instruction": question, "input": "", "output": answer_text})
            split_data[2].append({"instruction": question + " " + description, "input": "", "output": answer_text})
            split_data[3].append({"instruction": question + " " + description, "input": "", "output": answer_text})
            split_data[4].append({"instruction": question, "input": description, "output": answer_text})
            split_data[5].append({"instruction": question, "input": description, "output": answer_text})
        else:
            split_data[0].append({"instruction": question, "input": "", "output": answer_text})
            split_data[2].append({"instruction": question + " " + description, "input": "", "output": answer_text})
            split_data[4].append({"instruction": question, "input": description, "output": answer_text})
        answer_count += 1
        
for i in range(6):
    with open("psyqa-"+str(i)+'.json', 'w', encoding='utf-8') as f:
        json.dump(split_data[i], f, ensure_ascii=False, indent=1)
