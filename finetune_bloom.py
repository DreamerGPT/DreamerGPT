import os
import sys

import torch
import torch.nn as nn
#import bitsandbytes as bnb
from datasets import load_dataset
import transformers
import random
import numpy as np
import argparse
from transformers import Trainer, TrainingArguments
from transformers import BloomForCausalLM, BloomTokenizerFast

from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)

from huggingface_hub import login, HfFolder

# login(
#   token="", # ADD YOUR TOKEN HERE
#   add_to_git_credential=True
# )

def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Argument and global variables
parser = argparse.ArgumentParser('Finetune4Bloom')
# parser.add_argument('-d', '--data', type=str, help='Dataset name', default='gowalla_Entertainment')
parser.add_argument('--bs', type=int, default=128, help='Batch_size')
parser.add_argument('--mbs', type=int, default=4, help='Micro_Batch_size')
parser.add_argument('--n_heads', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=3, help='Number of epochs')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
# parser.add_argument('--drop', type=float, default=0.5, help='Dropout')
# parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--lenco', type=int, default=256, help='Cut Off Len')
# parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
# parser.add_argument('--model', type=str, default="graphsage", choices=["graphsage", "sgc", "gcn", "gin", "gat", "dgi"], help='Type of embedding module')
# parser.add_argument('--n_hidden', type=int, default=256, help='Dimensions of the hidden')
# parser.add_argument("--fanout", type=str, default='15,10,5', help='Neighbor sampling fanout')
# parser.add_argument("--fanout_sgc", type=str, default='0', help='SGC neighbor sampling fanout')
# parser.add_argument('--different_new_nodes', action='store_true', help='Whether to use disjoint set of new nodes for train and val')
# parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
# parser.add_argument('--randomize_features', action='store_true', help='Whether to randomize node features')
# parser.add_argument('--data_type', type=str, default="gowalla", help='Type of dataset')
# parser.add_argument('--task_type', type=str, default="time_trans", help='Type of task')
parser.add_argument('--mode', type=str, default="pretrain", help='nopretrain, pretrain or downstream')
parser.add_argument('--seed', type=int, default=1234, help='Seed for all')
# parser.add_argument('--k_hop', type=int, default=3, help='K-hop for SGC')
# parser.add_argument('--learn_eps', action="store_true", help='learn the epsilon weighting')
# parser.add_argument('--aggr_type', type=str, default="mean", choices=["sum", "mean", "max"], help='type of neighboring pooling: sum, mean or max')
# parser.add_argument('--dgi_lam', type=float, default=1., help='coefficient of dgi loss')
parser.add_argument('--data_path', type=str, default="./data/Belle_open_source_0.5M.json", help='Path of Data')  # "./"
parser.add_argument('--model_path', type=str, default="./bloom-7b1", help='Path of Model')   # "./"
parser.add_argument('--out_path', type=str, default="./", help='Path of Log')
parser.add_argument('--valsize', type=int, default=2000, help='Set Size of Val')
parser.add_argument('--RFC_path', type=str, help='Path of FromCheckpoint')

##LORA 
parser.add_argument('--lorar', type=int, default=8, help='R of Lora')
parser.add_argument('--loraa', type=int, default=16, help='Alpha of Lora')
parser.add_argument('--lorad', type=float, default=0.05, help='Dropout of Lora')

args = parser.parse_args()
set_seed(args.seed)

MICRO_BATCH_SIZE = args.mbs  # this could actually be 5 but i like powers of 2
BATCH_SIZE = args.bs
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = args.n_epoch  # we don't always need 3 tbh
LEARNING_RATE = args.lr  # the Karpathy constant
CUTOFF_LEN = args.lenco  # 256 accounts for about 96% of the data
LORA_R = args.lorar
LORA_ALPHA = args.loraa
LORA_DROPOUT = args.lorad
VAL_SET_SIZE = args.valsize #2000
DATA_PATH = args.data_path #"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/xuhao19/DATA4HOPE/llm4sft/Belle_open_source_0.5M.json"
OUTPUT_DIR = args.out_path #"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/xuhao19/HOPE4OUTPUT/llmsft/bloom_lora/v2"
if args.RFC_path:
    resume_from_checkpoint = args.RFC_path
else:
    resume_from_checkpoint = False

model_name = args.model_path #'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/xuhao19/model/bloom/bloom-7b1'


device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    GRADIENT_ACCUMULATION_STEPS = GRADIENT_ACCUMULATION_STEPS // world_size

model = BloomForCausalLM.from_pretrained( 
    model_name,
    device_map=device_map,
    load_in_8bit=True,
)
tokenizer = BloomTokenizerFast.from_pretrained(model_name)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
data = load_dataset("json", data_files=DATA_PATH)


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
{data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
{data_point["output"]}"""


def tokenize(prompt):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )
    return {
        "input_ids": result["input_ids"][:-1],
        "attention_mask": result["attention_mask"][:-1],
    }


def generate_and_tokenize_prompt(data_point):
    # This function masks out the labels for the input,
    # so that our loss is computed only on the response.
    user_prompt = (
        (
            f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Input:
{data_point["input"]}

### Response:
"""
        )
        if data_point["input"]
        else (
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{data_point["instruction"]}

### Response:
"""
        )
    )
    len_user_prompt_tokens = (
        len(
            tokenizer(
                user_prompt,
                truncation=True,
                max_length=CUTOFF_LEN + 1,
                padding="max_length",
            )["input_ids"]
        )
        - 1
    )  # no eos token
    full_tokens = tokenizer(
        user_prompt + data_point["output"],
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )["input_ids"][:-1]
    return {
        "input_ids": full_tokens,
        "labels": [-100] * len_user_prompt_tokens
        + full_tokens[len_user_prompt_tokens:],
        "attention_mask": [1] * (len(full_tokens)),
    }


if VAL_SET_SIZE > 0:
    train_val = data["train"].train_test_split(
        test_size=VAL_SET_SIZE, shuffle=True, seed=42
    )
    train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
else:
    train_data = data['train'].shuffle().map(generate_and_tokenize_prompt)
    val_data = None

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps" if VAL_SET_SIZE > 0 else "no",
        save_strategy="steps",
        eval_steps=200 if VAL_SET_SIZE > 0 else None,
        save_steps=200,
        output_dir=OUTPUT_DIR, 
        save_total_limit=3,
        load_best_model_at_end=True if VAL_SET_SIZE > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        # torch_compile=True, # optimizations
        optim="adamw_torch", # improved optimizer，pytorch>2.0
        # push to hub parameters
        report_to=None,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

if torch.__version__ >= "2" and sys.platform != 'win32':
    model = torch.compile(model)

if resume_from_checkpoint:
    # Check the available weights and load them
    checkpoint_name = os.path.join(
        resume_from_checkpoint, "pytorch_model.bin"
    )  # Full checkpoint
    if not os.path.exists(checkpoint_name):
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "adapter_model.bin"
        )  # only LoRA model - LoRA config above has to fit
        resume_from_checkpoint = (
            False  # So the trainer won't try loading its state
        )
    # The two files above have a different name depending on how they were saved, but are actually the same.
    if os.path.exists(checkpoint_name):
        print(f"Restarting from {checkpoint_name}")
        adapters_weights = torch.load(checkpoint_name)
        set_peft_model_state_dict(model, adapters_weights)
    else:
        print(f"Checkpoint {checkpoint_name} not found")
# If you want to resume a training phase, please choose 'True'
# Else choose 'False'
trainer.train(resume_from_checkpoint = resume_from_checkpoint)

model.save_pretrained(OUTPUT_DIR)

# trainer.create_model_card()
# trainer.push_to_hub()

print("\n If there's a warning about missing keys above, please disregard :)")