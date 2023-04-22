# psyQA

[https://github.com/thu-coai/PsyQA](https://github.com/thu-coai/PsyQA)

划分：
```python
python split.py
```

| 内容 | 数据 | size |
|--|--|--|
|1个question（作为instruction）对应多个answer（作为output），不考虑description | psyqa-0.json |  91M |
|1个question（作为instruction）只对应其中的一个answer（作为output），不考虑description | psyqa-1.json | 38M |
|1个question（作为instruction）对应多个answer（作为output），考虑description，把其文本拼接到question后面共同作为instruction | 	psyqa-2.json | 123M |
|1个question（作为instruction）只对应其中的一个answer（作为output），考虑description，把其文本拼接到question后面共同作为instruction | 	psyqa-3.json | 49M |
|1个question（作为instruction）对应多个answer（作为output），考虑description（作为input）	| psyqa-4.json | 123M |
|1个question（作为instruction）只对应其中的一个answer（作为output），考虑description（作为input）| 	psyqa-5.json | 49M |
