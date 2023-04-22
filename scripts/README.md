# Bash Scripts

**脚本命名规则**：例如 `run-1-alpaca-13b-1.sh`

- run代表热启动（仅在f1），rerun代表增量训练，generate表示推理
- 1代表实验序号为1（表格第一行）
- alpaca-13b代表是从alpaca-lora 13b模型开始热启动的
- 最后的1代表第一列，比如f2，最后一位就是2，如果还有分支可以继续添加，比如2-1
- 注意，填写output路径时和脚本名称一致（方便查看结果），**不需要手动创建路径**

- rerun的脚本自己写的时候注意加上 `--resume_from_checkpoint '前一次执行的权重路径’`
- 不需要inputs loss，修改 `--train_on_inputs False`

## 实验规划

<!-- | start | f1 | f2 | f3 |
|--|--|--|--|
| Chinese-alpaca-lora-13b-热启动, 实验序号：1 | 数据：GPT4英文 `scripts/run-1-alpaca-13b-1.sh` | 数据：流萤-train-0 `scripts/rerun-1-alpaca-13b-1.sh` |  |
| Chinese-alpaca-lora-13b-热启动, 实验序号：2 | 数据：流萤-train-0 `scripts/run-2-alpaca-13b-1.sh` |  |  |
| Chinese-alpaca-lora-7b-热启动, 实验序号：3 | 数据：GPT4英文 `scripts/run-3-alpaca-7b-1.sh` |  |  |
| Chinese-alpaca-lora-7b-热启动, 实验序号：4 | 数据：流萤-train-0 `scripts/run-4-alpaca-7b-1` |  |  | -->