# CUDA_VISIBLE_DEVICES=0,1,2,3
cd "$(dirname $0)"
export PATH="/usr/local/conda/envs/lora/bin:$PATH"
python ../finetune.py \
    --base_model '/path/to/model/chinese-alpaca-lora-13b' \
    --data_path '/path/to/data/COIG/translated_instructions.jsonl' \
    --output_dir '/path/to/output/rerun-2-alpaca-13b-2-2' \
    --batch_size 256 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 500 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length \
    --resume_from_checkpoint '/path/to/output/rerun-2-alpaca-13b-2'