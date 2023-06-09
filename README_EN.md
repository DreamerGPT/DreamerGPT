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

## <img src="assets/project.png" style="vertical-align: middle; width: 35px;"> 1. Introduction of DreamerGPT

**The goal of this project is to promote the application of Chinese large language models in more vertical application fields.**

The following is a 8b quantitative demo. Inference acceleration and performance optimization are also being iterated:

![demo2](./assets/demo2.gif)

---

## <img src="assets/update.png" style="vertical-align: middle; width: 35px;"> 2. Recent Updates

<img src="assets/new.png" style="vertical-align: middle; width: 20px;">**[2023/04/23]** Officially open source of **DreamerGPT**, currently v0.1 has been provided for download experience!

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

### 3.2 Data

Data download links and pre-processing scripts:
| Data                                                     | Type          |
| ------------------------------------------------------------ | ---------------- |
| [Alpaca-GPT4](https://github.com/tloen/alpaca-lora/blob/main/alpaca_data_gpt4.json) | English           |
| [Firefly](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/firefly) | Chinese             |
| [COIG](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/COIG) | Chinese, English, code |
| [PsyQA](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/psyQA) | Chinese, psychological counseling    |
| [BELLE](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/belle) | Chinese            |
| [baize](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/baize) | Chinese, dialogue         |
| [Couplets](https://github.com/DreamerGPT/DreamerGPT/tree/main/data/couplets) | Chinese            |

**NOTE**: The datasets come from the open source community and can be accessed through the link.

---

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
| Chinese-alpaca-lora-13b-热启动 | Data：firefly-train-0 | Data：COIG-part1，COIG-translate | Data：PsyQA-5 |

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

---

## <img src="assets/test.png" style="vertical-align: middle; width: 35px;"> 6. Test Report

<a name="Test"></a>

There are currently 8 categories of test tasks for the evaluation samples (numerical ethics and Duolun Dialogue to be evaluated), with 10 samples in each category, and are scored according to GPT3.5/GPT-4, and the scoring range for each sample is 0-10. See `test_data/` for evaluation samples.

### 6.1 Testing Prompt
```
Below are the outputs of five ChatGPT-like systems. Please rate each item on a 10-point scale and provide an explanation to justify your score. The output format is: System score; System explanation.
Prompt:xxxx.
Answer:
System1:xxxx.
System2:xxxx.
System3:xxxx.
System4:xxxx.
System5:xxxx.
```

### 6.2 Scoring by GPT-4
**NOTE**: Scoring is for reference only. From the evalution results, GPT-4's scoring is more accurate compared with GPT-3.5.

| Test Task     | Detail Samples                               | Number of samples | D13b-1-3-1 | D13b-2-2-2 | D13b-2-3 | D7b-4-1 | ChatGPT  |
| ------------ | ------------------------------------------------------------ | ------ | ---------- | ---------- | -------- | ------- | -------- |
| Total score for each  category  | ---                                                          | 80     | 100        | 100        | 100      | 100     | 100      |
| Quiz     | [01qa.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/01qa.json) | 10     | 80*        | 78         | 78       | 68      | **95**   |
| Translation     | [02translate.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/02translate.json) | 10     | 77*        | 77*        | 77*      | 64      | **86**   |
| Text Generation  | [03generate.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/03generate.json) | 10     | 56         | 65*        | 55       | 61      | **91**   |
| Sentiment Analysis     | [04analyse.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/04analyse.json) | 10     | **91**     | **91**     | **91**   | 88*     | 88*      |
| Reading Comprehension   | [05understanding.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/05understanding.json) | 10     | 74*        | 74*        | 74*      | 76.5    | **96.5** |
| Chinese Characteristics   | [06chinese.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/06chinese.json) | 10     | 69*        | 69*        | 69*      | 43      | **86**   |
| Code Generation  | [07code.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/07code.json) | 10     | 62*        | 62*        | 62*      | 57      | **96**   |
| Ethics & Refusal | [08alignment.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/08alignment.json) | 10     | 87*        | 87*        | 87*      | 71      | **95.5** |
| Mathematical Calculations   | (To be evaluted)                                                  | --     | --         | --         | --       | --      | --       |
| Multi-rounds Dialogue    | (To be evaluted)                                   | --     | --         | --         | --       | --      | --       |

### 6.3 Scoring by GPT-3.5

| Test Task     | Detail Samples                                 | Number of samples | D13b-1-3-1 | D13b-2-2-2 | D13b-2-3 | D7b-4-1 | ChatGPT |
| ------------ | ------------------------------------------------------------ | ------ | ---------- | ---------- | -------- | ------- | ------- |
| Total score for each  category  | ---                                                          | 80     | 100        | 100        | 100      | 100     | 100     |
| Quiz     | [01qa.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/01qa.json) | 10     | 65         | 64         | 63       | 67*     | **89**  |
| Translation     | [02translate.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/02translate.json) | 10     | 79         | 81         | 82       | 89*     | **91**  |
| Text Generation    | [03generate.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/03generate.json) | 10     | 65         | 73*        | 63       | 71      | **92**  |
| Sentiment Analysis | [04analyse.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/04analyse.json) | 10     | 88*        | **91**     | 88*      | 85      | 71      |
| Reading Comprehension | [05understanding.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/05understanding.json) | 10     | 75         | 77         | 76       | 85*     | **91**  |
| Chinese Characteristics | [06chinese.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/06chinese.json) | 10     | 82*        | **83**     | 82*      | 40      | 68      |
| Code Generation     | [07code.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/07code.json) | 10     | 72         | 74         | 75*      | 73      | **96**  |
| Ethics & Refusal | [08alignment.json](https://github.com/DreamerGPT/DreamerGPT/blob/main/test_data/08alignment.json) | 10     | 71*        | 70         | 67       | 71*     | **94**  |
| Mathematical Calculations  | (To be evaluted)                                                   |   --     |     --       |     --       |     --    |     --    |     --    |
| Multi-rounds Dialogue   | (To be evaluted)                                                   |   --     |       --     |      --      |    --      |   --      |    --     |

Generally, the model has a good performance in **Translation**, **Sentiment Analysis**, and **Reading Comprehension**.

---

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

---

## <img src="assets/contact.png" style="vertical-align: middle; width: 35px;"> Contact us

There are still many deficiencies in this project, feel free to give us more suggestions, we will try our best to improve this project.

Emails：dreamergpt@gmail.com
