# couplets

[https://huggingface.co/shibing624/songnet-base-chinese-couplet](https://huggingface.co/shibing624/songnet-base-chinese-couplet)

```python
python split.py
```

数据示例：
|Instruction |	Output |
|--|--|
|晚风摇树树还挺	| 晨露润花花更红 |
|新居落成创业始 |	宏图初振治家先 |
|腾飞上铁，锐意改革谋发展，勇当千里马 |	和谐南供，安全送电保畅通，争做领头羊 |

数据构成：
| 内容 |	文件 | size |
|--|--|--|
|已对齐格式的741,096条对联韵律数据，原SongNet的训练集 |	couplet-0.json | 81M |
|已对齐格式的3,834条对联韵律数据，原SongNet的测试集 |	couplet-1.json | 427K |
|已对齐格式的744,930条对联韵律数据，全集数据（训练+测试） |	couplet-all.json | 81M |