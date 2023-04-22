# 梦想家 (DreamerGPT)

Project 梦想家 (DreamerGPT) was found by [迟慧璇](https://github.com/ytchx1999)，[徐灏](https://github.com/KingsleyHsu)，[贝元琛](https://github.com/YuanchenBei)，[刘丹阳](https://github.com/danyang-liu)。

## 1、项目介绍

中文大模型指令精调。

## 2、最近更新

[2023/04/23] 正式开源中文Alpaca-LoRA指令精调大模型----**梦想家（DreamerGPT）**，目前提供xxx版本下载体验

已开源模型：

## 3、模型和数据准备

### 3.1 模型

模型权重下载：

- [llama原始权重](https://github.com/facebookresearch/llama) 
- [chinese-llama/alpaca-lora权重](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

### 3.2 数据

数据下载以及预处理脚本。统一处理成以下json格式：

```bash
{
    "instruction": "...",
    "input": "...",
    "output": "..."
}
```

| 数据                                   | 类型             |
| -------------------------------------- | ---------------- |
| [Alpaca-GPT4](#)                       | 英文             |
| [Firefly (预处理成多份，格式对齐)](#)  | 中文             |
| [COIG](#)                              | 中文、代码、中英 |
| [PsyQA (预处理成多份，格式对齐)](#)    | 中文心理咨询     |
| [BELLE](#)                             | 中文             |
| [baize](#)                             | 中文对话         |
| [Couplets (预处理成多份，格式对齐)](#) | 中文             |

## 4、训练代码和脚本

代码和脚本介绍：

- `finetune.py`：指令精调热启动/增量训练代码
- `generate.py`：推理/测试代码
- `scripts/`：运行脚本
  - 比如：`scripts/rerun-2-alpaca-13b-2.sh`，各参数解释见`scripts/README.md`

## 5、如何使用

### 5.1 环境安装

详细信息和相关问题请参考[Alpaca-LoRA](https://github.com/tloen/alpaca-lora)。

```bash
pip install -r requirements.txt
```

### 5.2模型权重合并

权重融合（以alpaca-lora-13b为例）：

```bash
cd scripts/
bash merge-13b-alpaca.sh
```

参数含义（请自行修改相关路径）：

- `--base_model`, llama原始权重
- `--lora_model`, chinese-llama/alpaca-lora权重 
- `--output_dir`, 输出融合权重的路径

### 5.3 指令微调

以下面的训练流程为例，展示运行的脚本。

| start                                       | f1                 | f2                               | f3            |
| ------------------------------------------- | ------------------ | -------------------------------- | ------------- |
| Chinese-alpaca-lora-13b-热启动，实验序号：2 | 数据：流萤-train-0 | 数据：COIG-part1，COIG-translate | 数据：PsyQA-5 |

```bash
cd scripts/
# 热启动f1
bash run-2-alpaca-13b-1.sh
# 增量训练f2
bash rerun-2-alpaca-13b-2.sh
bash rerun-2-alpaca-13b-2-2.sh
# 增量训练f3
bash rerun-2-alpaca-13b-3.sh
```

重要参数解释（请自行修改相关路径）：

- 基础路径信息请参考[Alpaca-LoRA](https://github.com/tloen/alpaca-lora)。
- rerun的脚本自己写的时候注意加上 `--resume_from_checkpoint '前一次执行的LoRA权重路径’`
- 不需要inputs loss，修改 `--train_on_inputs False`
- 测试集的大小 `--val_set_size 2000` ，如果数据集本身就比较小，可适当减小，比如500， 200

### 5.4 推理/测评

比如，我要测评`rerun-2-alpaca-13b-2.sh`微调后的结果：

1、网页版交互：

```bash
cd scripts/
bash generate-2-alpaca-13b-2.sh
```

2、批量推理并保存结果：

```bash
cd scripts/
bash save-generate-2-alpaca-13b-2.sh
```

重要参数解释（请自行修改相关路径）：

- `--is_ui False `：是否是网页版，默认为True
-  `--test_input_path 'xxx.json' `：输入的instruction路径
- 输出结果默认保存在对应LoRA权重目录下的`test.json`中

## 6、评测报告

come soon!

## 7、下一版更新内容

- 长文本数据
- 多轮对话数据

## 局限性、使⽤限制与免责声明



## 引⽤



## 联系我们

