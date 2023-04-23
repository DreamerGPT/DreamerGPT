# <img src="assets/hummingbird.png" style="vertical-align: middle; width: 68px;"> DreamerGPT: Chinese Instruction-tuning for Large Language Model.

🌱 **DreamerGPT is a project of *Chinese Instruction-tuning for Large Language Model*, found by [Hao Xu](https://github.com/KingsleyHsu), [Huixuan Chi](https://github.com/ytchx1999), [Yuanchen Bei](https://github.com/YuanchenBei) and [Danyang Liu](https://github.com/danyang-liu).**

*👉 Read in [Chinese version](README.md)*.

<div align=center>
<img src="assets/climb.jpg" style="vertical-align: middle; width: 550px;">
</div> 
<br>
<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/DreamerGPT/DreamerGPT">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/DreamerGPT/DreamerGPT">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/DreamerGPT/DreamerGPT">
    <img alt="Stars" src="https://img.shields.io/github/stars/DreamerGPT/DreamerGPT?color=yellow">
</p>

---

## <img src="assets/project.png" style="vertical-align: middle; width: 35px;"> 1. Introduction

**DreamerGPT**: A Chinese instruction enhanced large language model.

---

## <img src="assets/update.png" style="vertical-align: middle; width: 35px;"> 2. Recent Updates

**[2023/04/23]** Officially open source of **DreamerGPT**, currently v0.1 has been provided for download experience!

Existing models (continuous incremental training, more models to be updated in the future):

| Model   |  Data  | Weight Path  | Download Link  |
| ---------- | ------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------ |
| D13b-1-3-1 | Chinese-alpaca-lora-13b-热启动 + COIG-part1、COIG-translate + PsyQA-5 | `output/rerun-1-alpaca-13b-3-1/` | [Google Drive](https://drive.google.com/file/d/1PKT32_IMaHyE2qdt_W40Y6wxk3HVIma-/view?usp=sharing) |
| D13b-2-2-2 | Chinese-alpaca-lora-13b-热启动 + firefly-train-0 + COIG-part1、COIG-translate | `output/rerun-2-alpaca-13b-2-2/` | [Google Drive](https://drive.google.com/file/d/1WgzzKbc6IatBiHCcaQA5K74Y8fTEckSs/view?usp=sharing) |
| D13b-2-3   | Chinese-alpaca-lora-13b-热启动 + firefly-train-0 + COIG-part1、COIG-translate + PsyQA-5 | `output/rerun-2-alpaca-13b-3/`   | [Google Drive](https://drive.google.com/file/d/1sM2qNJcz0K43Y-MmhDXvw3hfqOtvzeNI/view?usp=sharing) |
| D7b-4-1    | Chinese-alpaca-lora-7b-热启动 + firefly-train-0              | `output/run-4-alpaca-7b-1/`      | [Google Drive](https://drive.google.com/file/d/1EAzMpgYA7nQ-9XR4NH4iwAtp83UIH3Bv/view?usp=sharing) |

[Evaluation results](#Test)

---

## <img src="assets/model.png" style="vertical-align: middle; width: 35px;"> 3. Model and Data Description

### 3.1 Model
- [Original weights of llama](https://github.com/facebookresearch/llama) 
- [Original weights of chinese-llama/alpaca-lora](https://github.com/ymcui/Chinese-LLaMA-Alpaca)

The data is uniformly pre-processed into the following *json* format:
```bash
{
    "instruction": "...",
    "input": "...",
    "output": "..."
}
```


Data download links and pre-processing scripts:
| Data                                                     | Type          |
| ------------------------------------------------------------ | ---------------- |
| [Alpaca-GPT4](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data_gpt4.json) | English           |
| [Firefly (预处理成多份，格式对齐)](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/firefly) | Chinese             |
| [COIG](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/COIG) | Chinese, English, code |
| [PsyQA (预处理成多份，格式对齐)](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/psyQA) | Chinese, psychological counseling    |
| [BELLE](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/belle) | Chinese            |
| [baize](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/baize) | Chinese, dialogue         |
| [Couplets (预处理成多份，格式对齐)](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/couplets) | Chinese            |

## <img src="assets/train.png" style="vertical-align: middle; width: 35px;"> 4. Training

Description of code and script:

- `finetune.py`: warm-start of instruction fine-tuning & incremental training
- `generate.py`：inference & test
- `scripts/`: runing scripts
  - e.g. `scripts/rerun-2-alpaca-13b-2.sh`, for the details of script parameters, see `scripts/README.md`.
  

## <img src="assets/handbook.png" style="vertical-align: middle; width: 35px;"> 5. How to use

### 5.1 Requirements
For details and related questions please refer to [Alpaca-LoRA](https://github.com/tloen/alpaca-lora)。

```bash
pip install -r requirements.txt
```

### 5.2 Model Weight Merging

An example of the weight merging of alpaca-lora-13b:

```bash
cd scripts/
bash merge-13b-alpaca.sh
```

Parameter explanation:

- `--base_model`: original weights of llama
- `--lora_model`: original weights of chinese-llama/alpaca-lora 
- `--output_dir`: output path of the merged model weights

### 5.3 Instruction fine-tuning (optional)


**NOTE that** if you want to directly download the fine-tuned weights for inference, you can ignore *Sec 5.3* and go directly to *Sec 5.4*.

Take the following training process as an example to show the running script.

| start                                       | f1                    | f2                               | f3            |
| ------------------------------------------- | --------------------- | -------------------------------- | ------------- |
| Chinese-alpaca-lora-13b-热启动, Experiment ID: 2 | Data：firefly-train-0 | Data：COIG-part1，COIG-translate | Data：PsyQA-5 |

```bash
cd scripts/
# warm-start (f1)
bash run-2-alpaca-13b-1.sh
# incremental training (f2)
bash rerun-2-alpaca-13b-2.sh
bash rerun-2-alpaca-13b-2-2.sh
# incremental training (f3)
bash rerun-2-alpaca-13b-3.sh
```

Explanation of the important parameters:

- For basic path information, please refer to [Alpaca-LoRA](https://github.com/tloen/alpaca-lora).
- Pay attention to adding the rerun script when you make a new code writing.
- `--resume_from_checkpoint 'The LoRA weight path of the previous execution'`
- If you don't need the **inputs** loss, revise `--train_on_inputs False`
- The size of the test set `--val_set_size 2000`, If the used dataset is relatively small, it can be appropriately reduced, such as 500, 200.

### 5.4 Inference/Test

For example, if you want to evaluate the fine-tuned results after running `rerun-2-alpaca-13b-2.sh`:

(i) Chat with DreamerGPT online:

```bash
cd scripts/
bash generate-2-alpaca-13b-2.sh
```

(ii) Batch inference and save the results:

```bash
cd scripts/
bash save-generate-2-alpaca-13b-2.sh
```

Explanation of the important parameters:

- `--is_ui False `: whether to chat with DreamerGPT in the online manner, the default is **True**.
-  `--test_input_path 'xxx.json' `: the input instruction path.
- The output results are saved in `test.json` in the corresponding LoRA weight directory by default.

## <img src="assets/test.png" style="vertical-align: middle; width: 35px;"> 6. Test Report

<a name="Test"></a>

There are currently 8 categories of test tasks for the evaluation samples (numerical ethics and Duolun Dialogue to be evaluated), with 10 samples in each category, and are scored according to GPT-4/ChatGPT, and the scoring range for each sample is 0-10. See `test_data/` for evaluation samples.

| Test Task     | Detail Samples                                 | Number of samples | D13b-1-3-1 | D13b-2-2-2 | D13b-2-3 | D7b-4-1 | ChatGPT |
| ------------ | ------------------------------------------------------------ | ------ | ---------- | ---------- | -------- | ------- | ------- |
| Total score for each  category  | ---                                                          | 80     | 100        | 100        | 100      | 100     | 100     |
| Quiz     | [01qa.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/01qa.json) | 10     | 65         | 64         | 63       | 67*     | **89**  |
| **Translation**     | [02translate.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/02translate.json) | 10     | 79         | 81         | 82       | 89*     | **91**  |
| Text Generation    | [03generate.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/03generate.json) | 10     | 65         | 73*        | 63       | 71      | **92**  |
| **Sentiment Analysis** | [04analyse.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/04analyse.json) | 10     | 88*        | **91**     | 88*      | 85      | 71      |
| **Reading Comprehension** | [05understanding.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/05understanding.json) | 10     | 75         | 77         | 76       | 85*     | **91**  |
| **Chinese Characteristics** | [06chinese.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/06chinese.json) | 10     | 82*        | **83**     | 82*      | 40      | 68      |
| Code Generation     | [07code.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/07code.json) | 10     | 72         | 74         | 75*      | 73      | **96**  |
| Ethics & Refusal | [08alignment.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/08alignment.json) | 10     | 71*        | 70         | 67       | 71*     | **94**  |
| Numeral Calculations  | (To be evaluted)                                                   |   --     |     --       |     --       |     --    |     --    |     --    |
| Multiple Rounds of Dialogue   | (To be evaluted)                                                   |   --     |       --     |      --      |    --      |   --      |    --     |

The model has a good performance in **Translation**, **Sentiment Analysis**, **Reading Comprehension**, **Chinese Characteristics**.

## <img src="assets/cite.png" style="vertical-align: middle; width: 35px;"> Citation
```
@misc{DreamerGPT,
  author = {Hao Xu, Huixuan Chi, Yuanchen Bei and Danyang Liu},
  title = {DreamerGPT: Chinese Instruction-tuning for Large Language Model.},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DreamerGPT/DreamerGPT}},
}
```
